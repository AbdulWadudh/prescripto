"""
Indian Medical Prescription Analyzer — dual model, fully in-process.

  Prescriptions (image)  ->  Qwen2.5-VL 7B Instruct (Q4_K_M) + mmproj
  Follow-up / text       ->  Gemma 4-E4B-it (Q4_K_M)

Three analysis modes:
  Patient Wise  -> output/patients/{PatientName}/{YYYY-MM-DD_HHMMSS}/
  Doctor Wise   -> output/doctors/{DoctorName}/{YYYY-MM-DD_HHMMSS}/
  General       -> output/general/{YYYY-MM-DD_HHMMSS}_{Doctor}_{Patient}/

All results are saved to prescriptions.db (SQLite) at the project root.
"""

import os
import re
import json
import base64
import shutil
import mimetypes
import datetime
import sqlite3
import random
import gradio as gr
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

# ======================================================================
# CONFIG
# ======================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH    = os.path.join(BASE_DIR, "prescriptions.db")

TEXT_MODEL_PATH    = os.path.join(MODELS_DIR, "gemma-4-E4B-it-Q4_K_M.gguf")
VISION_MODEL_PATH  = os.path.join(MODELS_DIR, "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
VISION_MMPROJ_PATH = os.path.join(MODELS_DIR, "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")

N_GPU_LAYERS = -1
N_CTX_TEXT   = 8192
N_CTX_VISION = 8192
MAX_TOKENS   = 3072
TEMPERATURE  = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================================
# SQLITE — init + helpers
# ======================================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                mode              TEXT    NOT NULL,
                analyzed_at       TEXT    NOT NULL,
                image_filename    TEXT,
                output_dir        TEXT,
                doctor_name       TEXT,
                doctor_quals      TEXT,
                doctor_specialty  TEXT,
                doctor_clinic     TEXT,
                doctor_address    TEXT,
                doctor_contact    TEXT,
                doctor_reg_no     TEXT,
                patient_name      TEXT,
                patient_age       TEXT,
                patient_gender    TEXT,
                prescription_date TEXT,
                languages         TEXT,
                diagnosis         TEXT,
                medications       TEXT,
                investigations    TEXT,
                summary           TEXT,
                confidence_notes  TEXT,
                raw_json          TEXT    NOT NULL
            )
        """)
        conn.commit()

init_db()


def save_to_db(mode: str, image_filename: str, output_dir: str, data: dict) -> int:
    doc  = data.get("doctor")  or {}
    pat  = data.get("patient") or {}
    clin = data.get("clinical") or {}
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO prescriptions (
                mode, analyzed_at, image_filename, output_dir,
                doctor_name, doctor_quals, doctor_specialty, doctor_clinic,
                doctor_address, doctor_contact, doctor_reg_no,
                patient_name, patient_age, patient_gender, prescription_date,
                languages, diagnosis, medications, investigations,
                summary, confidence_notes, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mode,
            datetime.datetime.now().isoformat(timespec="seconds"),
            image_filename,
            output_dir,
            doc.get("name"),          doc.get("qualifications"), doc.get("specialty"),
            doc.get("clinic_hospital"), doc.get("address"),      doc.get("contact"),
            doc.get("registration_number"),
            pat.get("name"), pat.get("age"), pat.get("gender"), pat.get("date"),
            json.dumps(data.get("languages_detected") or [], ensure_ascii=False),
            json.dumps(clin.get("diagnosis")          or [], ensure_ascii=False),
            json.dumps(data.get("medications")        or [], ensure_ascii=False),
            json.dumps(data.get("investigations")     or [], ensure_ascii=False),
            data.get("summary"),
            data.get("confidence_notes"),
            json.dumps(data, indent=2, ensure_ascii=False),
        ))
        conn.commit()
        return cur.lastrowid

# ======================================================================
# PROMPTS
# ======================================================================
PRESCRIPTION_SYSTEM_PROMPT = """You are an expert medical prescription analyzer specializing in Indian healthcare documentation. Analyze handwritten or printed medical prescriptions that may be in any Indian language or a combination: Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, English, or any regional mix.

## Your Task
1. Read and transcribe every piece of visible text in the prescription exactly as written — store this verbatim (original script) ONLY in the "raw_transcription" field.
2. For EVERY other field in the JSON output, provide the value in English. Follow this format rule:
   - For fields containing non-English script text: write the English transliteration/translation first, then the original script in parentheses. Format: "English rendering (original script)". Example for a diagnosis word only: "Fever (बुखार)".
   - For pure English abbreviations: expand to full English, no brackets needed. Example: "F" → "Female", "BD" → "Twice Daily".
   - For fields that are already in English: leave as-is, no brackets.
3. Intelligently interpret abbreviations and shorthand used by Indian doctors.
4. Extract and categorize all information. Layouts vary — letterheads, stamps, and handwritten notes can appear anywhere on the page.
5. CRITICAL — null means absent. If a field's information is NOT clearly readable or printed in the image, the value MUST be null. Never use your training knowledge to fill in details. Forbidden behaviours:
   - Inventing a doctor name, qualification, specialty, or clinic because the letterhead is partial or you recognise the institution
   - Guessing patient name, age, or gender when not written
   - Adding medications, dosages, or diagnoses you cannot directly read
   When in doubt, set to null and note it in confidence_notes.

## Abbreviation Reference
**Patient:** M / F = Male / Female | 18/M or 18Y/M or 18Yr/M = 18-year-old Male | Y/O = Years Old | Pt = Patient
**Frequency:** OD = Once Daily | BD / BID = Twice Daily | TDS / TID = Three Times Daily | QID = Four Times Daily | HS = At Bedtime | AC = Before Meals | PC = After Meals | SOS / PRN = As Needed | Stat = Immediately | ON = Every Night
**Duration:** × or x = For (×5D = for 5 days) | D = Days | W = Weeks | M = Months
**Dosage Forms:** Tab = Tablet | Cap = Capsule | Syp / Syr = Syrup | Inj = Injection | Oint / Ung = Ointment | Susp = Suspension | Gt / Gtt = Drops | Sachet | Cream | Gel | Patch
**Clinical:** Rx = Prescription | c/o = Complains of | k/c/o = Known case of | h/o = History of | O/E = On Examination | D/D = Differential Diagnosis | B/P = Blood Pressure | PR = Pulse Rate | SPO2 = Oxygen Saturation | Wt = Weight | Ht = Height | T = Temperature
**Qualifications:** MBBS | MD | MS | DNB | DM | MCh | DGO | DCH | FRCS | BDS | MDS | BAMS | BHMS | BPT | MPT

## Output
Return ONLY a valid JSON object. No explanation, no markdown fences, no extra text — just the raw JSON.
Fill all structured fields first. raw_transcription comes last and must be a plain-text summary of key visible text only — maximum 80 words, no repetition:

{
  "languages_detected": [],
  "doctor": {
    "name": null,
    "qualifications": null,
    "designation": null,
    "specialty": null,
    "clinic_hospital": null,
    "address": null,
    "contact": null,
    "registration_number": null,
    "timing": null
  },
  "patient": {
    "name": null,
    "age": null,
    "gender": null,
    "date": null,
    "patient_id": null,
    "contact": null,
    "address": null
  },
  "clinical": {
    "chief_complaints": [],
    "history": [],
    "diagnosis": [],
    "vitals": {}
  },
  "medications": [
    {
      "name": null,
      "type": null,
      "dose": null,
      "frequency": null,
      "duration": null,
      "instructions": null
    }
  ],
  "investigations": [],
  "advice": [],
  "follow_up": null,
  "summary": null,
  "confidence_notes": null,
  "english_transcription": "full English translation of all visible text — prose, max 120 words",
  "raw_transcription": "verbatim copy in original script — max 80 words, no repetition"
}

For "english_transcription": write a complete, readable English rendering of everything visible — doctor header, patient details, all medications with instructions, any handwritten notes or stamps.
For "raw_transcription": exact verbatim copy of key text as it appears in the original language/script, no translation.
For the "summary" field: write 4–5 sentences covering the full clinical picture — who the patient is, what condition is being treated, each medication prescribed with its dosage and purpose, and any key advice or follow-up instructions. Be specific and explanatory, not generic."""

# Mode label → internal key
MODE_MAP = {
    "Patient Wise": "patient",
    "Doctor Wise":  "doctor",
    "General":      "general",
}

# ======================================================================
# LOAD MODELS
# ======================================================================
print(f"[load] text model:   {os.path.basename(TEXT_MODEL_PATH)}")
text_llm = Llama(
    model_path=TEXT_MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX_TEXT,
    verbose=False,
)

print(f"[load] vision model: {os.path.basename(VISION_MODEL_PATH)}")
vision_llm = None
try:
    vision_handler = Qwen25VLChatHandler(clip_model_path=VISION_MMPROJ_PATH, verbose=False)
    _vlm = Llama(
        model_path=VISION_MODEL_PATH,
        chat_handler=vision_handler,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX_VISION,
        logits_all=True,
        verbose=False,
    )
    # Probe mtmd init (lazy in >=0.3.x); needs >=2x2 image
    import struct as _struct, zlib as _zlib
    def _probe_uri():
        def _chunk(tag, data):
            return (_struct.pack('>I', len(data)) + tag + data
                    + _struct.pack('>I', _zlib.crc32(tag + data) & 0xFFFFFFFF))
        raw = b''.join(b'\x00' + b'\x80\x80\x80' * 4 for _ in range(4))
        png = (b'\x89PNG\r\n\x1a\n'
               + _chunk(b'IHDR', _struct.pack('>IIBBBBB', 4, 4, 8, 2, 0, 0, 0))
               + _chunk(b'IDAT', _zlib.compress(raw))
               + _chunk(b'IEND', b''))
        return "data:image/png;base64," + base64.b64encode(png).decode()
    _vlm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _probe_uri()}},
            {"type": "text", "text": "hi"},
        ]}],
        max_tokens=1, stream=False,
    )
    vision_llm = _vlm
    print("[load] both models ready")
except Exception as _exc:
    print(f"[warn] vision disabled — {_exc}")
    print("[warn] fix: pip install \"llama-cpp-python<0.3.0\"")
    print("[load] text model ready (vision disabled)")

# ======================================================================
# FILE HELPERS
# ======================================================================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


_MAX_IMG_DIM = 1280  # cap longest side; keeps token count manageable


def _normalize_image(img) -> bytes:
    from PIL import Image
    import io
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_IMG_DIM:
        scale = _MAX_IMG_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_data_uri(path: str) -> str:
    from PIL import Image
    with Image.open(path) as img:
        data = _normalize_image(img)
    return "data:image/png;base64," + base64.b64encode(data).decode()


def pdf_first_page_to_data_uri(path: str) -> str:
    import fitz
    from PIL import Image
    with fitz.open(path) as doc:
        pix = doc[0].get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = _normalize_image(img)
    return "data:image/png;base64," + base64.b64encode(data).decode()


def get_prescription_file(files: list) -> tuple[str | None, str | None]:
    """Return (file_path, data_uri) for the first image or PDF in the upload list."""
    for f in files or []:
        path = f if isinstance(f, str) else (f.get("path") or f.get("name"))
        if not path:
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in IMG_EXTS:
            return path, image_to_data_uri(path)
        if ext == ".pdf":
            return path, pdf_first_page_to_data_uri(path)
    return None, None


def slugify(name: str, maxlen: int = 25) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip()).strip("_")
    return s[:maxlen] if s else "unknown"

# ======================================================================
# PRESCRIPTION OUTPUT HELPERS
# ======================================================================

def extract_json_from_response(text: str) -> dict | None:
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def json_to_markdown(data: dict, image_filename: str = "", mode: str = "general") -> str:
    d = data
    now        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_label = {"patient": "Patient Wise", "doctor": "Doctor Wise",
                  "general": "General"}.get(mode, "General")

    lines = [
        "# Medical Prescription Analysis",
        f"*Mode: {mode_label}  ·  Analyzed: {now}*",
    ]
    if image_filename:
        lines.append(f"*Source: `{image_filename}`*")
    langs = d.get("languages_detected") or []
    if langs:
        lines.append(f"*Languages: {', '.join(langs)}*")
    lines.append("")

    def section(title: str, items: list):
        if items:
            lines.append(f"## {title}")
            lines.extend(f"- {i}" for i in items)
            lines.append("")

    def field(label: str, value):
        if value:
            lines.append(f"**{label}:** {value}")

    def doctor_block():
        doc = d.get("doctor") or {}
        if any(v for v in doc.values() if v):
            lines.append("## Doctor")
            field("Name",            doc.get("name"))
            field("Qualifications",  doc.get("qualifications"))
            field("Designation",     doc.get("designation"))
            field("Specialty",       doc.get("specialty"))
            field("Clinic / Hospital", doc.get("clinic_hospital"))
            field("Address",         doc.get("address"))
            field("Contact",         doc.get("contact"))
            field("Reg. No.",        doc.get("registration_number"))
            field("Timing",          doc.get("timing"))
            lines.append("")

    def patient_block():
        pat = d.get("patient") or {}
        if any(v for v in pat.values() if v):
            lines.append("## Patient")
            field("Name",       pat.get("name"))
            field("Age",        pat.get("age"))
            field("Gender",     pat.get("gender"))
            field("Date",       pat.get("date"))
            field("Patient ID", pat.get("patient_id"))
            field("Contact",    pat.get("contact"))
            field("Address",    pat.get("address"))
            lines.append("")

    def clinical_block():
        clin = d.get("clinical") or {}
        section("Chief Complaints", clin.get("chief_complaints") or [])
        section("History",          clin.get("history")          or [])
        section("Diagnosis",        clin.get("diagnosis")        or [])
        vitals = clin.get("vitals") or {}
        if vitals:
            lines.append("## Vitals")
            for k, v in vitals.items():
                lines.append(f"**{k}:** {v}")
            lines.append("")

    def medications_block():
        meds = [m for m in (d.get("medications") or []) if any(v for v in m.values() if v)]
        if meds:
            lines.append("## Medications")
            lines.append("| # | Medicine | Type | Dose | Frequency | Duration | Instructions |")
            lines.append("|---|---|---|---|---|---|---|")
            for i, m in enumerate(meds, 1):
                row = [str(i),
                       m.get("name")         or "—",
                       m.get("type")         or "—",
                       m.get("dose")         or "—",
                       m.get("frequency")    or "—",
                       m.get("duration")     or "—",
                       m.get("instructions") or "—"]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Fixed order: Doctor → Patient → Clinical → Medications → Investigations → Advice → Follow-up → Raw → Notes → Summary
    doctor_block()
    patient_block()
    clinical_block()
    medications_block()

    section("Investigations / Tests", d.get("investigations") or [])
    section("Advice",                 d.get("advice")         or [])

    if d.get("follow_up"):
        lines.append(f"## Follow-up\n{d['follow_up']}\n")
    if d.get("english_transcription"):
        lines.append(f"## English Transcription\n{d['english_transcription']}\n")
    if d.get("raw_transcription"):
        lines.append(f"## Original Text (Verbatim)\n```\n{d['raw_transcription']}\n```\n")
    if d.get("confidence_notes"):
        lines.append(f"## Confidence Notes\n*{d['confidence_notes']}*\n")
    if d.get("summary"):
        lines.append(f"## Summary\n{d['summary']}\n")

    return "\n".join(lines)


def save_prescription_output(image_path: str, json_data: dict,
                              md_text: str, mode: str) -> str:
    now      = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    doc_name = ((json_data.get("doctor")  or {}).get("name") or "")
    pat_name = ((json_data.get("patient") or {}).get("name") or "")

    if mode == "patient":
        out_dir = os.path.join(OUTPUT_DIR, "patients",
                               slugify(pat_name), f"{date_str}_{time_str}")
    elif mode == "doctor":
        out_dir = os.path.join(OUTPUT_DIR, "doctors",
                               slugify(doc_name), f"{date_str}_{time_str}")
    else:  # general
        out_dir = os.path.join(OUTPUT_DIR, "general",
                               f"{date_str}_{time_str}_{slugify(doc_name,15)}_{slugify(pat_name,15)}")

    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    shutil.copy2(image_path, os.path.join(out_dir, f"prescription{ext}"))
    with open(os.path.join(out_dir, "prescription.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "prescription.md"), "w", encoding="utf-8") as f:
        f.write(md_text)
    return out_dir

# ======================================================================
# HISTORY HELPER
# ======================================================================

def normalize_history(history):
    out = []
    for turn in history or []:
        if isinstance(turn, dict):
            role    = turn.get("role")
            content = turn.get("content", "")
            if isinstance(content, (list, tuple)):
                content = "\n".join(c for c in content if isinstance(c, str))
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, asst_msg = turn
            if user_msg:
                out.append({"role": "user",      "content": str(user_msg)})
            if asst_msg:
                out.append({"role": "assistant", "content": str(asst_msg)})
    return out

# ======================================================================
# CHAT CALLBACK
# ======================================================================

def respond(message, history, mode_label: str):
    mode = MODE_MAP.get(mode_label, "general")

    if isinstance(message, dict):
        user_text = message.get("text", "") or ""
        files     = message.get("files", []) or []
    else:
        user_text, files = str(message), []

    file_path, data_uri = get_prescription_file(files)

    # ---- prescription analysis (image or PDF uploaded) ----
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
            rec_id  = save_to_db(mode, fname, out_dir, json_data)
            rel     = os.path.relpath(out_dir).replace("\\", "/")
            # yield md_text + f"\n\n---\n*Saved to `{rel}`  ·  DB record #{rec_id}*"
            yield md_text
        else:
            yield (
                "Could not parse structured data from the model response.\n\n"
                f"**Raw output:**\n```\n{raw[:3000]}\n```"
            )
        return

    # ---- text follow-up ----
    print("[route: Gemma text]")
    messages = []
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": user_text or "Hello."})

    stream = text_llm.create_chat_completion(
        messages=messages, stream=True, temperature=0.7, max_tokens=MAX_TOKENS,
    )
    acc = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            acc += delta
            yield acc

# ======================================================================
# UI HELPERS
# ======================================================================
_MODE_COLOR = {"patient": "#4ade80", "doctor": "#60a5fa", "general": "#fb923c"}

def history_html(mode_filter: str = "All") -> str:
    query = """SELECT id, analyzed_at, mode, patient_name, doctor_name, summary
               FROM prescriptions"""
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
        mode_l = r["mode"].replace("patient","Patient Wise").replace("doctor","Doctor Wise").replace("general","General")
        summ   = (r["summary"] or "")[:90] + ("…" if len(r["summary"] or "") > 90 else "")
        rid = r["id"]
        cards.append(f"""
<div data-action="load:{rid}" class="rx-card" style="padding:10px 12px;margin-bottom:8px;background:#1e1e1e;border-radius:8px;border-left:3px solid {color};cursor:pointer">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px">
    <div style="font-size:10px;color:#555">#{rid} · {date}</div>
    <div>
      <button data-action="edit:{rid}" style="background:none;border:1px solid #444;color:#aaa;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:11px;margin-right:2px" title="Edit">✎</button>
      <button data-action="del:{rid}" style="background:none;border:1px solid #553333;color:#e55;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:11px" title="Delete">✕</button>
    </div>
  </div>
  <div style="font-weight:600;color:#e5e5e5;font-size:13px;margin-bottom:2px">{pt}</div>
  <div style="font-size:11px;color:#888;margin-bottom:6px">{doc}</div>
  <span style="font-size:10px;background:{color}22;color:{color};padding:2px 8px;border-radius:10px">{mode_l}</span>
  {"<div style='font-size:11px;color:#666;margin-top:6px'>"+summ+"</div>" if summ else ""}
</div>""")
    return "".join(cards)


def handle_action(action: str, mode_filter: str):
    """Handle load:/edit:/del: signals from sidebar card buttons."""
    action = (action or "").strip()
    _nc    = gr.update()  # no-op chatbot update
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


_SETUP_JS = """
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
      if (!bus) {
        console.warn('[rx] #action-bus not found');
        return;
      }
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

_SIDEBAR_CSS = """
#sidebar-col { border-right: 1px solid #2a2a2a; padding-right: 8px; }
#hist-scroll  { overflow-y: auto; max-height: calc(100vh - 200px); padding-right: 4px; }
#main-col { display: flex; flex-direction: column; }
#main-col .gradio-chatinterface { flex: 1 1 auto; min-height: 0; }
.rx-card:hover { background: #252525 !important; }
#action-bus { display: none; }
"""

# ======================================================================
# UI
# ======================================================================
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
            hist_out = gr.HTML(elem_id="hist-scroll")
            action_bus = gr.Textbox(elem_id="action-bus", container=False)
            with gr.Group(visible=False) as edit_panel:
                edit_id_state = gr.State(0)
                gr.Markdown("#### Edit Record")
                edit_patient = gr.Textbox(label="Patient Name")
                edit_doctor  = gr.Textbox(label="Doctor Name")
                edit_diag    = gr.Textbox(label="Diagnosis")
                edit_summary = gr.Textbox(label="Summary", lines=3)
                with gr.Row():
                    save_edit_btn   = gr.Button("Save", variant="primary", size="sm")
                    cancel_edit_btn = gr.Button("Cancel", size="sm")

        # ── Main area ────────────────────────────────────────────────
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
            )
            _chatbot = gr.Chatbot(
                height=600,
                label="",
            )
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
        outputs=[hist_out, edit_panel, edit_id_state, edit_patient, edit_doctor, edit_diag, edit_summary, _chatbot],
    )
    save_edit_btn.click(
        do_save_edit,
        inputs=[edit_id_state, edit_patient, edit_doctor, edit_diag, edit_summary, hist_filter],
        outputs=[hist_out, edit_panel],
    )
    cancel_edit_btn.click(lambda: gr.update(visible=False), outputs=edit_panel)

if __name__ == "__main__":
    prescripto.launch(css=_SIDEBAR_CSS, js=_SETUP_JS)
