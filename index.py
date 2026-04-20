"""
Local chatbot — dual model, fully in-process (no APIs, no servers).

  Text + PDFs  ->  Gemma 4-E4B-it (Q4_K_M)
  Images       ->  Qwen2.5-VL 7B Instruct (Q4_K_M) + mmproj

Both models are loaded once at startup and routed per message:
  - if the user's message contains an image  -> LLaVA
  - otherwise                                -> Gemma

UI:       Gradio (multimodal ChatInterface)
Runtime:  llama-cpp-python  (loads .gguf directly, no HTTP)
PDFs:     PyMuPDF extracts text, prepended to the user message

------------------------------------------------------------
INSTALL
------------------------------------------------------------
    pip install llama-cpp-python gradio pymupdf pillow

    # Optional GPU builds (Windows / NVIDIA), run in the same PowerShell:
    #   $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    #   pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

------------------------------------------------------------
RUN
------------------------------------------------------------
    python local_chatbot.py

------------------------------------------------------------
MEMORY NOTE
------------------------------------------------------------
Both models stay loaded: ~4.7 GB (Gemma) + ~4.3 GB (LLaVA) + ~0.6 GB (mmproj)
= roughly 10 GB of RAM / VRAM. If you're tight, set N_GPU_LAYERS = 0 to keep
everything on CPU/RAM, or comment out one of the two models.
"""

import os
import base64
import mimetypes
import gradio as gr
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

# ======================================================================
# CONFIG — paths and parameters
# ======================================================================
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

TEXT_MODEL_PATH    = os.path.join(MODELS_DIR, "gemma-4-E4B-it-Q4_K_M.gguf")
VISION_MODEL_PATH  = os.path.join(MODELS_DIR, "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
VISION_MMPROJ_PATH = os.path.join(MODELS_DIR, "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")

N_GPU_LAYERS = -1        # -1 = all layers on GPU; 0 = pure CPU
N_CTX_TEXT   = 8192
N_CTX_VISION = 4096      # vision images eat tokens, keep this modest
MAX_TOKENS   = 1024
TEMPERATURE  = 0.7

SYSTEM_PROMPT_TEXT = (
    "You are a helpful local assistant. The user may attach PDFs; their "
    "extracted text will be prepended to the user's message inside "
    "[PDF: ...] blocks. Answer clearly and refer to the attachments when "
    "relevant."
)

SYSTEM_PROMPT_VISION = (
    "You are a helpful local assistant that can see images. Describe and "
    "answer questions about the images the user attaches. If extra text "
    "context is included, use it too."
)

# ======================================================================
# LOAD BOTH MODELS
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
    # llama-cpp-python >=0.3.x initialises the mtmd (multimodal) context lazily on
    # the first call — probe it now so we fail fast at startup rather than crashing
    # mid-conversation. mtmd requires images >= 2x2; build a minimal 4x4 PNG via stdlib.
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
    _PROBE_URI = _probe_uri()
    _vlm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _PROBE_URI}},
            {"type": "text", "text": "hi"},
        ]}],
        max_tokens=1,
        stream=False,
    )
    vision_llm = _vlm
    print("[load] both models ready")
except Exception as _exc:
    print(f"[warn] vision disabled — {_exc}")
    print("[warn] your mmproj is the old CLIP format, incompatible with llama-cpp-python 0.3.x")
    print("[warn] fix: pip install \"llama-cpp-python<0.3.0\"  (CPU)  or rebuild with CUDA flags")
    print("[load] text model ready (vision disabled)")

# ======================================================================
# FILE HANDLING
# ======================================================================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
PDF_EXTS = {".pdf"}


def extract_pdf_text(path: str) -> str:
    import fitz  # PyMuPDF
    with fitz.open(path) as doc:
        return "\n\n".join(page.get_text() for page in doc).strip()


def image_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def sort_attachments(files):
    """Split uploads into (pdf_text_blobs, image_paths, other_text_blobs)."""
    pdf_blobs, images, other_blobs = [], [], []
    for f in files or []:
        path = f if isinstance(f, str) else (f.get("path") or f.get("name"))
        if not path:
            continue
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext in PDF_EXTS:
            pdf_blobs.append(f"[PDF: {name}]\n{extract_pdf_text(path)}")
        elif ext in IMG_EXTS:
            images.append(path)
        else:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    other_blobs.append(f"[File: {name}]\n{fh.read()}")
            except Exception:
                other_blobs.append(f"[Skipped unsupported file: {name}]")
    return pdf_blobs, images, other_blobs


# ======================================================================
# HISTORY / MESSAGE BUILDING
# ======================================================================
def normalize_history(history):
    """Keep only plain-text turns so we don't re-ship past images."""
    out = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if isinstance(content, (list, tuple)):
            content = "\n".join(c for c in content if isinstance(c, str))
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def build_text_user_content(user_text, blobs):
    prefix = ("\n\n".join(blobs) + "\n\n") if blobs else ""
    return (prefix + (user_text or "")).strip() or "Please review the attachments."


def build_vision_user_content(user_text, images, blobs):
    """LLaVA expects a list of content parts: image_url + text."""
    parts = [{"type": "image_url",
              "image_url": {"url": image_to_data_uri(p)}} for p in images]
    prefix = ("\n\n".join(blobs) + "\n\n") if blobs else ""
    text = (prefix + (user_text or "")).strip() or "Describe the image(s)."
    parts.append({"type": "text", "text": text})
    return parts


# ======================================================================
# CHAT CALLBACK
# ======================================================================
def respond(message, history):
    if isinstance(message, dict):
        user_text = message.get("text", "") or ""
        files = message.get("files", []) or []
    else:
        user_text, files = str(message), []

    pdf_blobs, images, other_blobs = sort_attachments(files)
    blobs = pdf_blobs + other_blobs

    # ---- route ----
    if images:
        if vision_llm is None:
            yield (
                "**Vision is disabled** — the vision model failed to load at startup. "
                "Check the console for details."
            )
            return
        llm = vision_llm
        system = SYSTEM_PROMPT_VISION
        user_content = build_vision_user_content(user_text, images, blobs)
        route_tag = f"[route: LLaVA vision ({len(images)} img)]"
    else:
        llm = text_llm
        system = SYSTEM_PROMPT_TEXT
        user_content = build_text_user_content(user_text, blobs)
        route_tag = "[route: Gemma text]"

    print(route_tag)

    messages = [{"role": "system", "content": system}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": user_content})

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    acc = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            acc += delta
            yield acc


# ======================================================================
# UI
# ======================================================================
demo = gr.ChatInterface(
    fn=respond,
    multimodal=True,
    title="Local Chatbot — Gemma + LLaVA",
    description=(
        "**Text / PDF:** `gemma-4-E4B-it-Q4_K_M` &nbsp;·&nbsp; "
        "**Vision:** `Qwen2.5-VL-7B-Instruct-Q4_K_M`\n\n"
        "Drop PDFs or images into the textbox. The right model is chosen "
        "automatically based on whether you attached an image."
    ),
    textbox=gr.MultimodalTextbox(
        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
                    ".txt", ".md"],
        file_count="multiple",
        placeholder="Ask something, and/or attach a PDF or image…",
    ),
)

if __name__ == "__main__":
    demo.launch()