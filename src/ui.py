import os
import json
import random
import sqlite3
import gradio as gr

from .config import DB_PATH, MODE_MAP, MAX_TOKENS, TEMPERATURE
from .db import save_to_db
from .llm import text_llm, vision_llm
from .prescription import (
    PRESCRIPTION_SYSTEM_PROMPT,
    get_prescription_file,
    extract_json_from_response,
    json_to_markdown,
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


def respond(message, history, mode_label: str):
    mode = MODE_MAP.get(mode_label, "general")

    if isinstance(message, dict):
        user_text = message.get("text", "") or ""
        files     = message.get("files", []) or []
    else:
        user_text, files = str(message), []

    file_path, data_uri = get_prescription_file(files)

    # ── prescription analysis ──────────────────────────────────────────
    if file_path:
        if vision_llm is None:
            yield "**Vision is disabled** — check the console for details."
            return

        yield "*Generating...*"
        print(f"[route: prescription/{mode}] {os.path.basename(file_path)}")

        user_content = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text",      "text": user_text.strip() or "Analyze."},
        ]

        try:
            result = vision_llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": PRESCRIPTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                stream=False,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=random.randint(1, 2**31 - 1),
            )
            raw = result["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"[error] vision inference failed: {exc}")
            yield f"**Analysis failed:** {exc}"
            return

        print(f"[prescription] response: {len(raw)} chars")

        json_data = extract_json_from_response(raw)
        if json_data:
            fname   = os.path.basename(file_path)
            md_text = json_to_markdown(json_data, fname, mode)
            out_dir = save_prescription_output(file_path, json_data, md_text, mode)
            save_to_db(mode, fname, out_dir, json_data)
            yield md_text
        else:
            yield (
                "Could not parse structured data from the model response.\n\n"
                f"**Raw output:**\n```\n{raw[:3000]}\n```"
            )
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
            _chatbot = gr.Chatbot(height=600, label="")
            gr.ChatInterface(
                fn=respond,
                chatbot=_chatbot,
                multimodal=True,
                fill_height=True,
                additional_inputs=[mode],
                textbox=gr.MultimodalTextbox(
                    file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"],
                    file_count="single",
                    placeholder="Upload a prescription image or PDF — no text needed. Or type a follow-up question.",
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
