import os
import json
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import gradio as gr

from .config import DB_PATH, MODE_MAP, MAX_TOKENS, TEMPERATURE, N_CTX_VISION
from .db import save_to_db
from .llm import text_llm, vision_llm
from .prescription import (
    DEFAULT_PROMPT_FILE,
    PRESCRIPTION_SYSTEM_PROMPT,
    get_formatter_for,
    get_prescription_files,
    extract_json_from_response,
    json_to_markdown,
    list_prompt_files,
    load_prompt,
    pack_into_batches,
    save_prescription_output,
)

_MODE_COLOR = {"patient": "#4ade80", "doctor": "#60a5fa", "general": "#fb923c"}


# ── Chat helpers ───────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Flatten any Gradio content shape into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif "text" in c:
                    t = c.get("text", "") or ""
                    if t:
                        parts.append(t)
                    if c.get("files"):
                        parts.append("[prescription image]")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        t = content.get("text", "") or ""
        if content.get("files") and not t:
            return "[prescription image]"
        return t
    return ""


def normalize_history(history):
    out = []
    for turn in history or []:
        if isinstance(turn, dict):
            role    = turn.get("role")
            content = _extract_text(turn.get("content", ""))
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content[:4000]})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, asst_msg = turn
            if isinstance(user_msg, dict):
                user_text = user_msg.get("text", "") or ""
                files = user_msg.get("files", [])
                if files and not user_text:
                    user_text = "[prescription image uploaded for analysis]"
                elif files:
                    user_text = f"{user_text} [with uploaded file]"
            else:
                user_text = str(user_msg) if user_msg else ""
            if user_text:
                out.append({"role": "user", "content": user_text})
            if asst_msg:
                asst_text = str(asst_msg)
                if len(asst_text) > 4000:
                    asst_text = asst_text[:4000] + "\n\n[...truncated for context window...]"
                out.append({"role": "assistant", "content": asst_text})
    return out


_VISION_LOG_KEEP = 8  # lines of llama.cpp vision log to display in UI

_RE_ENCODE_DONE = re.compile(r"image slice encoded in (\d+) ms")
_RE_BATCH_DONE  = re.compile(r"image decoded \(batch (\d+)/(\d+)\) in (\d+) ms")
_RE_LOG_ALLOW   = re.compile(
    r"(encoding image|image slice encoded|decoding image batch|image decoded)"
)


def _summarize_vision_logs(lines: list[str]) -> dict:
    encoded = 0
    encode_ms = 0
    batches = 0
    batch_ms = 0
    for line in lines:
        m = _RE_ENCODE_DONE.search(line)
        if m:
            encoded += 1
            encode_ms += int(m.group(1))
            continue
        m = _RE_BATCH_DONE.search(line)
        if m:
            batches += 1
            batch_ms += int(m.group(3))
    return {
        "encoded":    encoded,
        "encode_s":   encode_ms / 1000.0,
        "batches":    batches,
        "batch_s":    batch_ms / 1000.0,
    }


def _vision_infer_with_logs(vision_llm, **kwargs):
    """Run vision_llm.create_chat_completion while streaming stderr back.

    Yields ('log', line), ('tick', None), then either ('result', completion)
    or ('error', exception). Captures fd 2 (stderr) via an OS-level pipe so
    llama.cpp's C++ image-encoding logs are visible to the caller in real time.
    """
    log_q: "queue.Queue[tuple[str, object]]" = queue.Queue()
    result_holder: dict = {}

    # Set up fd 2 → pipe redirect
    r, w = os.pipe()
    try:
        sys.stderr.flush()
    except Exception:
        pass
    orig_fd = os.dup(2)
    os.dup2(w, 2)
    os.close(w)  # fd 2 keeps the write end alive; closing orig_fd later drops it

    def reader():
        buf = b""
        try:
            while True:
                chunk = os.read(r, 4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        log_q.put(("log", text))
            if buf.strip():
                log_q.put(("log", buf.decode("utf-8", errors="replace").rstrip()))
        except OSError:
            pass

    def worker():
        try:
            result_holder["result"] = vision_llm.create_chat_completion(**kwargs)
        except Exception as exc:
            result_holder["error"] = exc

    reader_t = threading.Thread(target=reader, daemon=True)
    worker_t = threading.Thread(target=worker, daemon=True)
    reader_t.start()
    worker_t.start()

    try:
        # Drain logs while worker is running; tick if the queue is quiet so
        # the generator can update elapsed-time readouts.
        while worker_t.is_alive():
            try:
                kind, val = log_q.get(timeout=0.3)
                yield kind, val
            except queue.Empty:
                yield "tick", None
    finally:
        worker_t.join()
        # Restore stderr so the reader sees EOF and exits.
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(orig_fd, 2)
        os.close(orig_fd)
        reader_t.join(timeout=2.0)
        try:
            os.close(r)
        except OSError:
            pass
        # Drain anything buffered after the worker finished.
        while not log_q.empty():
            try:
                kind, val = log_q.get_nowait()
                if kind == "log":
                    yield "log", val
            except queue.Empty:
                break

    if "error" in result_holder:
        yield "error", result_holder["error"]
    else:
        yield "result", result_holder["result"]


def respond(message, history, mode_label: str, prompt_file: str = DEFAULT_PROMPT_FILE):
    mode = MODE_MAP.get(mode_label, "general")
    system_prompt = load_prompt(prompt_file) if prompt_file else PRESCRIPTION_SYSTEM_PROMPT

    if isinstance(message, dict):
        user_text = message.get("text", "") or ""
        files     = message.get("files", []) or []
    else:
        user_text, files = str(message), []

    t_start = time.monotonic()
    prescription_files = get_prescription_files(files)
    t_prep = time.monotonic() - t_start

    # ── prescription analysis ──────────────────────────────────────────
    if prescription_files:
        if vision_llm is None:
            yield "**Vision is disabled** — check the console for details."
            return

        is_default_prompt = (prompt_file or DEFAULT_PROMPT_FILE) == DEFAULT_PROMPT_FILE
        # dedupe source paths (a multi-page PDF shows up once per page)
        seen: set[str] = set()
        file_paths: list[str] = []
        for fp, _ in prescription_files:
            if fp not in seen:
                seen.add(fp)
                file_paths.append(fp)
        fnames      = [os.path.basename(fp) for fp in file_paths]
        total_pages = len(prescription_files)
        total_src   = len(file_paths)

        # ── batch planning ─────────────────────────────────────────────
        # Each 1120px image ≈ 1600 vision tokens. We pack images into batches
        # whose combined tokens + system prompt + output buffer fit in ctx.
        VISION_TOKENS_PER_IMAGE = 1600
        sys_tok_est = max(1, len(system_prompt) // 4)
        per_batch_budget = max(
            VISION_TOKENS_PER_IMAGE,
            N_CTX_VISION - sys_tok_est - MAX_TOKENS - 1024,
        )
        batches = pack_into_batches(prescription_files, VISION_TOKENS_PER_IMAGE, per_batch_budget)
        num_batches = len(batches)

        steps: list[str] = [
            f"✅ **Prepared** {total_src} file{'s' if total_src > 1 else ''}, "
            f"{total_pages} page{'s' if total_pages > 1 else ''} — `{t_prep:.1f}s`"
        ]
        if num_batches > 1:
            max_per_batch = max(len(b) for b in batches)
            steps.append(
                f"📦 **Split into {num_batches} batches** "
                f"(≤ {max_per_batch} images/batch to fit {N_CTX_VISION:,}-token context)"
            )

        def progress(extra: str = "") -> str:
            body = "### Processing\n" + "\n".join(f"- {s}" for s in steps)
            return body + (f"\n\n---\n\n{extra}" if extra else "")

        yield progress()
        print(f"[route: prescription/{mode}] {total_src} file(s) / {total_pages} page(s), "
              f"{num_batches} batch(es): {fnames} [prompt: {prompt_file}]")

        default_instruction = (
            "Analyze these prescriptions together as one case." if total_src > 1 else "Analyze."
        )

        # ── run vision inference per batch ─────────────────────────────
        batch_outputs: list[tuple[list[str], str]] = []   # [(paths_in_batch, raw_output), ...]
        t_vision_total = 0.0

        for bi, batch in enumerate(batches, 1):
            batch_seen: set[str] = set()
            batch_paths: list[str] = []
            for fp, _ in batch:
                if fp not in batch_seen:
                    batch_seen.add(fp)
                    batch_paths.append(fp)
            batch_pages = len(batch)
            batch_label_prefix = (
                f"Batch {bi}/{num_batches} · " if num_batches > 1 else ""
            )

            steps.append(
                f"⏳ **{batch_label_prefix}Vision analyzing** "
                f"{batch_pages} image{'s' if batch_pages > 1 else ''}..."
            )
            yield progress()

            user_content = [
                {"type": "image_url", "image_url": {"url": uri}} for _, uri in batch
            ]
            user_content.append({
                "type": "text",
                "text": user_text.strip() or default_instruction,
            })

            t_b_start = time.monotonic()
            vision_logs: list[str] = []
            vision_result = None
            vision_error  = None

            def _batch_step(elapsed: float) -> str:
                tail = "\n".join(vision_logs[-_VISION_LOG_KEEP:])
                summary = _summarize_vision_logs(vision_logs)
                sub = []
                if summary["encoded"]:
                    sub.append(
                        f"encoded {summary['encoded']} slice"
                        f"{'s' if summary['encoded'] > 1 else ''} "
                        f"(`{summary['encode_s']:.1f}s`)"
                    )
                if summary["batches"]:
                    sub.append(
                        f"decoded {summary['batches']} batch"
                        f"{'es' if summary['batches'] > 1 else ''} "
                        f"(`{summary['batch_s']:.1f}s`)"
                    )
                line = (
                    f"⏳ **{batch_label_prefix}Vision analyzing** "
                    f"{batch_pages} image{'s' if batch_pages > 1 else ''} — `{elapsed:.1f}s`"
                    + (f" · {' · '.join(sub)}" if sub else "")
                )
                if tail:
                    line += f"\n\n```\n{tail}\n```"
                return line

            for kind, val in _vision_infer_with_logs(
                vision_llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                stream=False,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=random.randint(1, 2**31 - 1),
            ):
                if kind == "log":
                    # Filter stderr noise — keep only image encode/decode lines.
                    if not _RE_LOG_ALLOW.search(val):
                        continue
                    vision_logs.append(val)
                    steps[-1] = _batch_step(time.monotonic() - t_b_start)
                    yield progress()
                elif kind == "tick":
                    steps[-1] = _batch_step(time.monotonic() - t_b_start)
                    yield progress()
                elif kind == "error":
                    vision_error = val
                elif kind == "result":
                    vision_result = val

            t_batch = time.monotonic() - t_b_start
            t_vision_total += t_batch

            if vision_error is not None:
                print(f"[batch {bi}/{num_batches}] failed: {vision_error}")
                steps[-1] = (
                    f"❌ **{batch_label_prefix}Vision failed** — `{t_batch:.1f}s`: "
                    f"{vision_error}"
                )
                yield progress()
                continue

            raw = vision_result["choices"][0]["message"]["content"]
            print(f"[batch {bi}/{num_batches}] response: {len(raw)} chars in {t_batch:.1f}s")
            summary = _summarize_vision_logs(vision_logs)
            extra_bits = []
            if summary["encoded"]:
                extra_bits.append(f"encoded {summary['encoded']} `{summary['encode_s']:.1f}s`")
            if summary["batches"]:
                extra_bits.append(f"decoded {summary['batches']} `{summary['batch_s']:.1f}s`")
            steps[-1] = (
                f"✅ **{batch_label_prefix}Vision complete** — `{t_batch:.1f}s` "
                f"({batch_pages} image{'s' if batch_pages > 1 else ''}, "
                f"{len(raw):,} chars"
                + (f" · {' · '.join(extra_bits)}" if extra_bits else "")
                + ")"
            )
            yield progress()
            batch_outputs.append((batch_paths, raw))

        if not batch_outputs:
            yield "**All batches failed.** See console for details."
            return

        def timing_footer(**extra) -> str:
            t_total = time.monotonic() - t_start
            parts = [f"Total: `{t_total:.1f}s`", f"Prep: `{t_prep:.1f}s`",
                     f"Vision: `{t_vision_total:.1f}s`"]
            if num_batches > 1:
                parts.append(f"Batches: `{num_batches}`")
            for k, v in extra.items():
                parts.append(f"{k}: `{v:.1f}s`")
            return "\n\n---\n*" + " · ".join(parts) + "*"

        def _batch_heading(bi: int) -> str:
            return f"## Batch {bi}/{num_batches}\n\n" if num_batches > 1 else ""

        # ── Non-default prompt → raw output or formatter-rendered ─────
        if not is_default_prompt:
            formatter_file = get_formatter_for(prompt_file)
            if formatter_file:
                print(f"[formatter] applying {formatter_file} to {len(batch_outputs)} batch(es)")
                formatter_prompt = load_prompt(formatter_file)
                md_sections: list[str] = []
                t_fmt_start = time.monotonic()

                for bi, (_, raw) in enumerate(batch_outputs, 1):
                    parsed = extract_json_from_response(raw)
                    if not parsed:
                        md_sections.append(
                            _batch_heading(bi) + f"```\n{raw[:3000]}\n```"
                        )
                        continue
                    steps.append(
                        f"⏳ **Formatting batch {bi}/{len(batch_outputs)}** "
                        f"with `{formatter_file}`..."
                    )
                    yield progress()
                    json_payload = json.dumps(parsed, indent=2, ensure_ascii=False)
                    t_sec_start = time.monotonic()
                    try:
                        stream = text_llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": formatter_prompt},
                                {"role": "user",   "content": json_payload},
                            ],
                            stream=True,
                            temperature=0.2,
                            max_tokens=MAX_TOKENS,
                        )
                        acc = ""
                        for chunk in stream:
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                acc += delta
                                elapsed = time.monotonic() - t_sec_start
                                steps[-1] = (
                                    f"⏳ **Formatting batch {bi}/{len(batch_outputs)}** "
                                    f"with `{formatter_file}` — `{elapsed:.1f}s`"
                                )
                                preview = "\n\n---\n\n".join(md_sections + [_batch_heading(bi) + acc])
                                yield progress() + "\n\n---\n\n" + preview
                        steps[-1] = (
                            f"✅ **Formatted batch {bi}/{len(batch_outputs)}** — "
                            f"`{time.monotonic() - t_sec_start:.1f}s`"
                        )
                        yield progress()
                        md_sections.append(_batch_heading(bi) + acc)
                    except Exception as exc:
                        print(f"[formatter] batch {bi} failed: {exc}")
                        steps[-1] = f"⚠️ **Formatter failed** on batch {bi}: {exc}"
                        md_sections.append(_batch_heading(bi) + f"```\n{raw[:3000]}\n```")

                t_format = time.monotonic() - t_fmt_start
                yield "\n\n---\n\n".join(md_sections) + timing_footer(Formatter=t_format)
                return

            # No formatter → show raw outputs (concatenated if multi-batch)
            raw_joined = "\n\n---\n\n".join(
                _batch_heading(bi) + raw for bi, (_, raw) in enumerate(batch_outputs, 1)
            )
            yield raw_joined + timing_footer()
            return

        # ── Default prompt → parse JSON, save, render markdown per batch ─
        t_save_start = time.monotonic()
        md_sections: list[str] = []
        for bi, (paths, raw) in enumerate(batch_outputs, 1):
            combined_fname = ", ".join(os.path.basename(p) for p in paths)
            json_data = extract_json_from_response(raw)
            if json_data:
                md_text = json_to_markdown(json_data, combined_fname, mode)
                out_dir = save_prescription_output(paths, json_data, md_text, mode)
                save_to_db(mode, combined_fname, out_dir, json_data)
                md_sections.append(_batch_heading(bi) + md_text)
            else:
                md_sections.append(
                    _batch_heading(bi)
                    + "Could not parse structured data from the model response.\n\n"
                    + f"**Raw output:**\n```\n{raw[:3000]}\n```"
                )
        t_save = time.monotonic() - t_save_start
        yield "\n\n---\n\n".join(md_sections) + timing_footer(Save=t_save)
        return

    # ── text follow-up ─────────────────────────────────────────────────
    print("[route: Gemma text]")
    messages = []
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": user_text or "Hello."})
    print(f"[context] {len(messages)} messages in context ({sum(len(m['content']) for m in messages)} chars)")

    stream = text_llm.create_chat_completion(
        messages=messages, stream=True, temperature=0.7, max_tokens=MAX_TOKENS,
    )
    acc = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            acc += delta
            yield acc


# ── Sidebar helpers ────────────────────────────────────────────────────

def history_html(mode_filter: str = "All") -> str:
    query  = "SELECT id, analyzed_at, mode, patient_name, doctor_name, summary FROM prescriptions"
    params = []
    if mode_filter and mode_filter != "All":
        params.append(mode_filter.lower().replace(" wise", ""))
        query += " WHERE mode = ?"
    query += " ORDER BY id DESC LIMIT 80"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return "<p style='color:#666;padding:16px;text-align:center;font-size:13px'>No records yet</p>"

    cards = []
    for r in rows:
        color  = _MODE_COLOR.get(r["mode"], "#888")
        date   = (r["analyzed_at"] or "")[:16]
        pt     = r["patient_name"] or "Unknown patient"
        doc    = r["doctor_name"]  or "Unknown doctor"
        mode_l = (r["mode"]
                  .replace("patient", "Patient Wise")
                  .replace("doctor",  "Doctor Wise")
                  .replace("general", "General"))
        summ = (r["summary"] or "")[:90] + ("…" if len(r["summary"] or "") > 90 else "")
        rid  = r["id"]
        cards.append(f"""
<div data-action="load:{rid}" class="rx-card" style="padding:10px 12px;margin-bottom:8px;background:#1e1e1e;border-radius:8px;border-left:3px solid {color};cursor:pointer">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px">
    <div style="font-size:10px;color:#555">#{rid} · {date}</div>
    <div>
      <button data-action="edit:{rid}" style="background:none;border:1px solid #444;color:#aaa;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:11px;margin-right:2px" title="Edit">✎</button>
      <button data-action="del:{rid}"  style="background:none;border:1px solid #553333;color:#e55;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:11px" title="Delete">✕</button>
    </div>
  </div>
  <div style="font-weight:600;color:#e5e5e5;font-size:13px;margin-bottom:2px">{pt}</div>
  <div style="font-size:11px;color:#888;margin-bottom:6px">{doc}</div>
  <span style="font-size:10px;background:{color}22;color:{color};padding:2px 8px;border-radius:10px">{mode_l}</span>
  {"<div style='font-size:11px;color:#666;margin-top:6px'>" + summ + "</div>" if summ else ""}
</div>""")
    return "".join(cards)


def handle_action(action: str, mode_filter: str):
    action = (action or "").strip()
    _nc    = gr.update()
    _empty = (history_html(mode_filter), gr.update(visible=False), 0, "", "", "", "", _nc)

    if action.startswith("del:"):
        try:
            rid = int(action[4:])
        except ValueError:
            return _empty
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM prescriptions WHERE id=?", (rid,))
        return history_html(mode_filter), gr.update(visible=False), 0, "", "", "", "", _nc

    if action.startswith("edit:"):
        try:
            rid = int(action[5:])
        except ValueError:
            return _empty
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM prescriptions WHERE id=?", (rid,)).fetchone()
        if not r:
            return _empty
        diag = ", ".join(json.loads(r["diagnosis"] or "[]"))
        return (
            history_html(mode_filter),
            gr.update(visible=True),
            rid,
            r["patient_name"] or "",
            r["doctor_name"]  or "",
            diag,
            r["summary"]      or "",
            _nc,
        )

    if action.startswith("load:"):
        try:
            rid = int(action[5:])
        except ValueError:
            return _empty
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM prescriptions WHERE id=?", (rid,)).fetchone()
        if not r:
            return _empty
        md_path = os.path.join(r["output_dir"] or "", "prescription.md")
        if r["output_dir"] and os.path.isfile(md_path):
            with open(md_path, encoding="utf-8") as f:
                md_content = f.read()
        else:
            try:
                data = json.loads(r["raw_json"])
                md_content = json_to_markdown(data, r["image_filename"] or "", r["mode"])
            except Exception:
                md_content = f"**Record #{rid}** — {r['patient_name']} / {r['doctor_name']}"
        note = (
            f"\n\n---\n*Loaded from history — Record #{rid} · "
            f"{(r['analyzed_at'] or '')[:16]}. "
            f"You can ask follow-up questions below.*"
        )
        return (
            history_html(mode_filter),
            gr.update(visible=False),
            0, "", "", "", "",
            gr.update(value=[[None, md_content + note]]),
        )

    return _empty


def do_save_edit(rid: int, patient_name: str, doctor_name: str,
                 diagnosis_str: str, summary: str, mode_filter: str):
    if rid:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE prescriptions SET patient_name=?, doctor_name=?, summary=? WHERE id=?",
                (patient_name.strip(), doctor_name.strip(), summary.strip(), rid),
            )
            row = conn.execute("SELECT raw_json FROM prescriptions WHERE id=?", (rid,)).fetchone()
            if row and row[0]:
                try:
                    data = json.loads(row[0])
                    data.setdefault("patient", {})["name"] = patient_name.strip()
                    data.setdefault("doctor",  {})["name"] = doctor_name.strip()
                    data["summary"] = summary.strip()
                    conn.execute(
                        "UPDATE prescriptions SET raw_json=? WHERE id=?",
                        (json.dumps(data, ensure_ascii=False), rid),
                    )
                except (json.JSONDecodeError, AttributeError):
                    pass
    return history_html(mode_filter), gr.update(visible=False)


# ── CSS / JS ───────────────────────────────────────────────────────────

SETUP_JS = """
() => {
  function rxSetup() {
    document.addEventListener('click', function(e) {
      var el = e.target.closest('[data-action]');
      if (!el) return;
      var action = el.getAttribute('data-action');
      if (action.startsWith('del:')) {
        if (!confirm('Delete record #' + action.slice(4) + '? This cannot be undone.')) return;
      }
      var c = document.getElementById('action-bus');
      var bus = c ? (c.querySelector('textarea') || c.querySelector('input')) : null;
      if (!bus) { console.warn('[rx] #action-bus not found'); return; }
      var nativeSet = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value') ||
                      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
      if (nativeSet && nativeSet.set) nativeSet.set.call(bus, action);
      else bus.value = action;
      bus.dispatchEvent(new Event('input',  {bubbles: true}));
      bus.dispatchEvent(new Event('change', {bubbles: true}));
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', rxSetup);
  } else {
    rxSetup();
  }
}
"""

SIDEBAR_CSS = """
#sidebar-col { border-right: 1px solid #2a2a2a; padding-right: 8px; }
#hist-scroll  { overflow-y: auto; max-height: calc(100vh - 200px); padding-right: 4px; }
#main-col { display: flex; flex-direction: column; }
#main-col .gradio-chatinterface { flex: 1 1 auto; min-height: 0; }
.rx-card:hover { background: #252525 !important; }
#action-bus { display: none; }
"""


# ── Gradio UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="Indian Medical Prescription Analyzer", fill_height=True) as prescripto:
    with gr.Row(equal_height=True):

        # ── Sidebar ──────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=270, elem_id="sidebar-col", visible=False):
            gr.Markdown("### Prescription History")
            hist_filter = gr.Dropdown(
                choices=["All", "Patient Wise", "Doctor Wise", "General"],
                value="All", label="Filter", container=False,
            )
            hist_refresh = gr.Button("↻  Refresh", size="sm", variant="secondary")
            hist_out     = gr.HTML(elem_id="hist-scroll")
            action_bus   = gr.Textbox(elem_id="action-bus", container=False)
            with gr.Group(visible=False) as edit_panel:
                edit_id_state = gr.State(0)
                gr.Markdown("#### Edit Record")
                edit_patient = gr.Textbox(label="Patient Name")
                edit_doctor  = gr.Textbox(label="Doctor Name")
                edit_diag    = gr.Textbox(label="Diagnosis")
                edit_summary = gr.Textbox(label="Summary", lines=3)
                with gr.Row():
                    save_edit_btn   = gr.Button("Save",   variant="primary", size="sm")
                    cancel_edit_btn = gr.Button("Cancel", size="sm")

        # ── Main area ─────────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="main-col"):
            gr.Markdown(
                "## Indian Medical Prescription Analyzer\n"
                "Upload a handwritten or printed prescription image or PDF. "
                "Supports all Indian languages — Hindi, Tamil, Telugu, Kannada, Malayalam, "
                "Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu — and mixed-language prescriptions."
            )
            mode = gr.Radio(
                choices=["Patient Wise", "Doctor Wise", "General"],
                value="General",
                label="Analysis Mode",
                info=(
                    "Patient Wise — output grouped under patient name  ·  "
                    "Doctor Wise — grouped under doctor / clinic  ·  "
                    "General — full combined analysis"
                ),
                visible=False,
            )
            _prompt_options = list_prompt_files() or [("Default Prompt", DEFAULT_PROMPT_FILE)]
            prompt_selector = gr.Dropdown(
                choices=_prompt_options,
                value=DEFAULT_PROMPT_FILE,
                label="System Prompt",
                info="Choose which prompt to send to the vision model. Add more by dropping .txt files in the prompts/ folder.",
                interactive=True,
            )
            _chatbot = gr.Chatbot(height=600, label="")
            gr.ChatInterface(
                fn=respond,
                chatbot=_chatbot,
                multimodal=True,
                fill_height=True,
                additional_inputs=[mode, prompt_selector],
                textbox=gr.MultimodalTextbox(
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"],
                    file_count="multiple",
                    placeholder="Upload one or more prescription images or PDFs. Or type a follow-up question.",
                ),
            )

    hist_filter.change(history_html, inputs=hist_filter, outputs=hist_out)
    hist_refresh.click(history_html, inputs=hist_filter, outputs=hist_out)
    prescripto.load(lambda: history_html("All"), outputs=hist_out)
    action_bus.change(
        handle_action,
        inputs=[action_bus, hist_filter],
        outputs=[hist_out, edit_panel, edit_id_state, edit_patient,
                 edit_doctor, edit_diag, edit_summary, _chatbot],
    )
    save_edit_btn.click(
        do_save_edit,
        inputs=[edit_id_state, edit_patient, edit_doctor, edit_diag, edit_summary, hist_filter],
        outputs=[hist_out, edit_panel],
    )
    cancel_edit_btn.click(lambda: gr.update(visible=False), outputs=edit_panel)
