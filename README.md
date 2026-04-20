# Local Chatbot — Setup & Run Guide

A fully offline, dual-model chatbot that runs entirely on your machine — no API keys, no internet connection required after setup. It uses two specialized models routed automatically per message.

| Input type | Model used |
|---|---|
| Text, PDFs, plain files | Gemma 4 E4B Instruct (Q4_K_M) |
| Images (with or without text) | Qwen2.5-VL 7B Instruct (Q4_K_M) + mmproj |

---

## How it works

When you send a message:
1. The script checks if any **image** was attached.
2. If yes → the message is routed to the **Qwen2.5-VL vision model**, which can see and reason about the image.
3. If no → the message is routed to **Gemma**, which handles text, PDFs, and file content.

Both models are loaded into memory once at startup and stay loaded for the entire session, so responses are fast after the first one.

PDFs are not sent as files — **PyMuPDF extracts the text** from them and prepends it to your message. This means the model reads the content directly rather than processing a binary file.

---

## Requirements

- Python 3.10 or later
- ~10 GB of free RAM or VRAM (see [Memory](#memory))
- Windows, macOS, or Linux

---

## Step 1 — Install Python dependencies

```bash
pip install llama-cpp-python gradio pymupdf pillow
```

**What each package does:**

| Package | Purpose |
|---|---|
| `llama-cpp-python` | Loads and runs `.gguf` model files directly in Python — no separate server needed |
| `gradio` | Provides the web UI (chat interface in your browser) |
| `pymupdf` | Extracts text from PDF files so the model can read them |
| `pillow` | Image utilities used internally by Gradio |

### GPU acceleration (Windows + NVIDIA only)

By default, `llama-cpp-python` runs on CPU. To enable CUDA (significantly faster):

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

This recompiles the package with CUDA support. You only need to do this once. Requires the NVIDIA CUDA Toolkit to be installed.

---

## Step 2 — Download the model files

Place all four files inside a `models/` folder next to `index.py`:

```
ai-gateway/
├── index.py
└── models/
    ├── gemma-4-E4B-it-Q4_K_M.gguf
    ├── Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
    └── mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf
```

The script resolves the `models/` path relative to its own location, so it works on any machine regardless of where the project is cloned.

### Text model
**Gemma 4 E4B Instruct Q4_K_M**
- File: `gemma-4-E4B-it-Q4_K_M.gguf`
- Size: ~4.7 GB
- Source: HuggingFace — search `bartowski/gemma-4-E4B-it-GGUF`

### Vision model (two files required)
**Qwen2.5-VL 7B Instruct Q4_K_M**
- File: `Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- Size: ~4.4 GB
- Source: HuggingFace — search `bartowski/Qwen2.5-VL-7B-Instruct-GGUF`

**Vision projector (mmproj)**
- File: `mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf`
- Size: ~1.3 GB
- Source: same repo as above

> The mmproj file is a separate encoder that maps image pixels into the token space the language model understands. It must match the vision model exactly — you cannot mix mmproj files from different models.

---

## Step 3 — Run the chatbot

```bash
python index.py
```

The script will print startup progress in the terminal:

```
[load] text model:   gemma-4-E4B-it-Q4_K_M.gguf
[load] vision model: Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
[load] both models ready
* Running on local URL:  http://127.0.0.1:7860
```

Open **http://127.0.0.1:7860** in your browser. The UI will show a chat box where you can type messages and attach files.

Loading both models takes 30–90 seconds depending on your hardware. Subsequent messages within the same session respond immediately.

---

## Step 4 — Using the chatbot

### Plain text
Type any question and press Enter. Gemma handles it.

### Attach a PDF
Click the paperclip icon, select a `.pdf`. The text is extracted and sent to Gemma with your question. Example:

> *"Summarise the key points from this document"* + attach `report.pdf`

### Attach an image
Attach any `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, or `.bmp`. The message is automatically routed to Qwen2.5-VL. Example:

> *"What does this chart show?"* + attach `graph.png`

### Attach both an image and a PDF
Both are processed — the PDF text is prepended as context and the image goes to the vision model together.

### Attach a text/code file
`.txt` and `.md` files are read as plain text and included in the Gemma context, same as PDFs.

---

## Configuration

All tuneable settings are at the top of [index.py](index.py):

```python
MODELS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
N_GPU_LAYERS  = -1      # -1 = all layers on GPU, 0 = CPU only
N_CTX_TEXT    = 8192    # context window for Gemma (tokens)
N_CTX_VISION  = 4096    # context window for Qwen-VL (smaller — images use many tokens)
MAX_TOKENS    = 1024    # maximum tokens per response
TEMPERATURE   = 0.7     # 0 = deterministic, 1 = more creative
```

**When to change these:**

- `N_GPU_LAYERS = 0` — if you have no GPU or run out of VRAM; runs on CPU/RAM instead (slower)
- `N_CTX_TEXT` — increase if you need to process very long PDFs; uses more VRAM
- `MAX_TOKENS` — increase for longer responses, decrease to speed things up
- `TEMPERATURE` — lower for factual/structured tasks, higher for creative writing

---

## Memory

Both models are loaded simultaneously:

| Component | Approx. size |
|---|---|
| Gemma 4 E4B (text) | ~4.7 GB |
| Qwen2.5-VL 7B (vision) | ~4.4 GB |
| Qwen mmproj (vision encoder) | ~1.3 GB |
| **Total** | **~10.4 GB** |

If you don't have enough VRAM/RAM:
- Set `N_GPU_LAYERS = 0` to offload everything to system RAM (slower but uses less VRAM)
- Or comment out the vision model block if you only need text/PDF support

---

## Troubleshooting

### `[warn] vision disabled` at startup
The vision model or mmproj failed to load. Check the console message for details. Common causes:
- Wrong mmproj file (must match the vision model)
- Insufficient VRAM to load both models with GPU layers

The chatbot will still run with text and PDF support — only image inputs will be disabled.

### Out of memory / crash during load
Set `N_GPU_LAYERS = 0` in the script to keep models on RAM instead of VRAM, or reduce `N_CTX_TEXT` / `N_CTX_VISION`.

### Slow first response
Normal — the model is warming up. Subsequent responses in the same session are faster.

### Port already in use
If `7860` is taken, Gradio will automatically try the next available port and print the URL in the terminal.
