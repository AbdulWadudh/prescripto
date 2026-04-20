import os
import re
import json
import base64
import shutil
import datetime
from json_repair import repair_json
from .config import BASE_DIR, OUTPUT_DIR, IMG_EXTS, PROMPTS_DIR

_MAX_IMG_DIM = 1120  # 28 * 40 — ~1600 img tokens, fits in 16k ctx
_PATCH      = 28    # Qwen2.5-VL patch size — dims must be multiples of this

# Load system prompt from prompts/default_prompt.txt at import time
_PROMPT_FILE = os.path.join(PROMPTS_DIR, "default_prompt.txt")
with open(_PROMPT_FILE, encoding="utf-8") as _f:
    PRESCRIPTION_SYSTEM_PROMPT = _f.read().strip()


# ── Image helpers ──────────────────────────────────────────────────────

def _normalize_image(img) -> bytes:
    from PIL import Image
    import io
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_IMG_DIM:
        scale = _MAX_IMG_DIM / max(w, h)
        w, h = int(w * scale), int(h * scale)
    # Dimensions must be multiples of the patch size or llama.cpp crashes
    w = max(_PATCH, round(w / _PATCH) * _PATCH)
    h = max(_PATCH, round(h / _PATCH) * _PATCH)
    img = img.resize((w, h), Image.LANCZOS)
    print(f"[image] resized to {w}x{h}")
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


def get_prescription_file(files: list) -> tuple:
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


# ── JSON / Markdown helpers ────────────────────────────────────────────

def extract_json_from_response(text: str) -> dict | None:
    print(f"[json] response length: {len(text)}")

    # Strip code fences
    cleaned = re.sub(r'`+(?:json)?', '', text).strip()

    try:
        result = json.loads(repair_json(cleaned))
        if isinstance(result, dict):
            print(f"[json] parse OK")
            return result
    except Exception as e:
        print(f"[json] failed: {e}")

    print(f"[json] all attempts failed")
    return None


def json_to_markdown(data: dict, image_filename: str = "", mode: str = "general") -> str:
    d          = data
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

    def section(title, items):
        if items:
            lines.append(f"## {title}")
            lines.extend(f"- {i}" for i in items)
            lines.append("")

    def field(label, value):
        if value:
            lines.append(f"**{label}:** {value}")

    def doctor_block():
        doc = d.get("doctor") or {}
        if any(v for v in doc.values() if v):
            lines.append("## Doctor")
            field("Name",              doc.get("name"))
            field("Qualifications",    doc.get("qualifications"))
            field("Designation",       doc.get("designation"))
            field("Specialty",         doc.get("specialty"))
            field("Clinic / Hospital", doc.get("clinic_hospital"))
            field("Address",           doc.get("address"))
            field("Contact",           doc.get("contact"))
            field("Reg. No.",          doc.get("registration_number"))
            field("Timing",            doc.get("timing"))
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

    doctor_block()
    patient_block()
    clinical_block()
    medications_block()
    section("Investigations / Tests", d.get("investigations") or [])
    section("Advice",                 d.get("advice")         or [])

    if d.get("follow_up"):
        lines.append(f"## Follow-up\n{d['follow_up']}\n")

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
    else:
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
