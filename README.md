# Indian Medical Prescription Analyzer

A fully offline AI-powered tool for analyzing handwritten and printed Indian medical prescriptions. No API keys, no internet connection required after setup. Two specialized models are loaded locally and routed automatically.

| Input | Model |
|---|---|
| Prescription image or PDF | Qwen2.5-VL 7B Instruct (vision) |
| Text follow-up questions | Gemma 4 E4B Instruct (text) |

---

## Features

- **Fully offline** — runs entirely on your machine after downloading the model files
- **All Indian languages** — Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, and mixed-language prescriptions
- **Structured extraction** — doctor info, patient details, medications table, diagnosis, vitals, investigations, advice, follow-up
- **Saved output** — every analysis saves the original image, structured JSON, and a formatted Markdown report to disk
---

## Project Structure

```
ai-gateway/
├── index.py                  ← entry point — run this
├── prescriptions.db          ← SQLite database (auto-created)
├── src/
│   ├── config.py             ← all paths and tunable constants
│   ├── db.py                 ← database schema and CRUD
│   ├── llm.py                ← model loading (Gemma + Qwen2.5-VL)
│   ├── prescription.py       ← image processing, JSON extraction, Markdown rendering
│   └── ui.py                 ← Gradio interface and chat callback
├── models/                   ← place GGUF model files here
│   ├── google_gemma-4-E4B-it-Q4_K_M.gguf
│   ├── Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
│   └── mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf
├── output/                   ← saved prescription analyses (auto-created)
│   ├── general/
│   ├── patients/
│   └── doctors/
└── prompts/
    ├── default_prompt.txt    ← vision model system prompt (edit freely)
    └── prescription_analysis.md  ← field reference and documentation
```

---

## Requirements

- Python 3.10 or later
- ~11 GB free VRAM (GPU) or RAM (CPU fallback)
- Windows, macOS, or Linux
- [uv](https://docs.astral.sh/uv/) — fast Python package and environment manager

---

## Step 1 — Install uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the install:

```bash
uv --version
```

---

## Step 2 — Create a virtual environment

Inside the project folder:

```bash
cd prescripto
uv venv
```

This creates a `.venv` folder. Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

---

## Step 3 — Install dependencies

Install all packages except `llama-cpp-python`:

```bash
uv pip install -r requirements.txt
```

### llama-cpp-python — CPU (any platform)

```bash
uv pip install llama-cpp-python
```

### llama-cpp-python — GPU / CUDA (NVIDIA, significantly faster)

Requires the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) to be installed first.

```bash
# Windows PowerShell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
uv pip install llama-cpp-python --no-cache-dir

# macOS / Linux
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
```

| Package | Purpose |
|---|---|
| `llama-cpp-python` | Loads and runs `.gguf` model files locally |
| `gradio` | Web UI in the browser |
| `pymupdf` | Renders the first page of PDF prescriptions as an image |
| `pillow` | Image resizing and format conversion |
| `json-repair` | Repairs malformed JSON output from the vision model |

---

## Step 4 — Download model files

Place all three files inside the `models/` folder:

### Text model — Gemma 4 E4B Instruct Q4_K_M
- **File:** `google_gemma-4-E4B-it-Q4_K_M.gguf`
- **Size:** ~4.7 GB
- **Download:** https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/resolve/main/google_gemma-4-E4B-it-Q4_K_M.gguf?download=true
- **Purpose:** Handles text follow-up questions after prescription analysis

### Vision model — Qwen2.5-VL 7B Instruct Q4_K_M
- **File:** `Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- **Size:** ~4.4 GB
- **Download:** https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf?download=true
- **Purpose:** Reads and extracts information from prescription images

### Vision projector (required for vision model)
- **File:** `mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf`
- **Size:** ~1.3 GB
- **Download:** https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF/blob/main/mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf
- **Purpose:** Encodes image pixels into tokens the language model understands. Must match the vision model exactly.

---

## Step 5 — Run

```bash
python index.py
```

Startup output:

```
[load] text model:   gemma-4-E4B-it-Q4_K_M.gguf
[load] vision model: Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
[load] both models ready
* Running on local URL:  http://127.0.0.1:7860
```

Open **http://127.0.0.1:7860** in your browser. Loading both models takes 30–90 seconds on first run.

---

## Step 6 — Using the analyzer

### Analyze a prescription
Upload a `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, or `.pdf` file using the attachment button. No text message is required — the model will analyze the prescription automatically. For PDFs, the first page is rendered and analyzed.

### Ask follow-up questions
After a prescription is analyzed, type any question in the chat. The full prescription context is retained, so questions like *"What is this medication used for?"* or *"Explain the dosage for the first medicine"* will be answered in context by Gemma.

### View prescription history
Past analyses are stored in the database. Click any record in the sidebar to load the full prescription report and continue asking questions from there.

### Edit a record
Click the pencil (✎) button on any history card to edit the patient name, doctor name, or summary inline.

### Delete a record
Click the ✕ button on any history card. A confirmation prompt will appear before deletion.

---

## Output

Every analyzed prescription is saved to the `output/` folder. The subfolder structure depends on the analysis mode:

```
output/
├── general/
│   └── 2026-04-20_143022_Dr_Sharma_Ravi_Kumar/
│       ├── prescription.jpg      ← original uploaded image
│       ├── prescription.json     ← full structured extraction
│       └── prescription.md       ← formatted Markdown report
├── patients/
│   └── Ravi_Kumar/
│       └── 2026-04-20_143022/
│           ├── prescription.jpg
│           ├── prescription.json
│           └── prescription.md
└── doctors/
    └── Dr_Sharma/
        └── 2026-04-20_143022/
            ├── prescription.jpg
            ├── prescription.json
            └── prescription.md
```

### Extracted fields

| Section | Fields |
|---|---|
| Doctor | Name, qualifications, designation, specialty, clinic/hospital, address, contact, registration number, timing |
| Patient | Name, age, gender, date, patient ID, contact, address |
| Clinical | Chief complaints, history, diagnosis, vitals |
| Medications | Name, type, dose, frequency, duration, instructions (table format) |
| Other | Investigations, advice, follow-up, summary, confidence notes, raw transcription |

All fields except `raw_transcription` are output in English. The `raw_transcription` field contains verbatim text in the original script as it appears on the prescription.

---

## Configuration

All settings are in `src/config.py`:

```python
N_GPU_LAYERS = -1       # -1 = all layers on GPU, 0 = CPU only
N_CTX_TEXT   = 8192     # context window for Gemma (tokens)
N_CTX_VISION = 16384    # context window for Qwen2.5-VL (tokens)
MAX_TOKENS   = 4096     # maximum output tokens per response
TEMPERATURE  = 0.1      # 0 = deterministic, 1 = creative
```

| Setting | When to change |
|---|---|
| `N_GPU_LAYERS = 0` | No GPU or insufficient VRAM — falls back to CPU/RAM (slower) |
| `N_CTX_TEXT` | Increase for longer follow-up conversations |
| `N_CTX_VISION` | Increase for very dense prescriptions; uses more VRAM |
| `MAX_TOKENS` | Increase if prescription analysis is getting cut off |
| `TEMPERATURE` | Keep low (0.1) for accurate extraction; raise for creative text tasks |

### Customizing the system prompt

The vision model's instructions are loaded from `prompts/default_prompt.txt` at startup. Edit this file to change extraction behavior, add abbreviations, or adjust output format. No Python changes required — just restart the app.

---

## Memory usage

| Component | Approx. VRAM/RAM |
|---|---|
| Gemma 4 E4B Q4_K_M | ~4.7 GB |
| Qwen2.5-VL 7B Q4_K_M | ~4.4 GB |
| mmproj encoder | ~1.3 GB |
| KV cache (16k vision ctx) | ~1.0 GB |
| **Total** | **~11.4 GB** |

If you run out of VRAM, set `N_GPU_LAYERS = 0` to offload to system RAM, or reduce `N_CTX_VISION` to `8192`.

---

## Troubleshooting

### `[warn] vision disabled` at startup
The vision model or mmproj failed to load. Text follow-ups still work; only prescription image analysis is disabled. Common causes:
- Wrong or missing mmproj file (must be from the same model repo)
- Insufficient VRAM — try reducing `N_CTX_VISION` or set `N_GPU_LAYERS = 0`

### `[error] vision inference failed: access violation`
Image dimensions were not aligned to the model's patch size. The app handles this automatically — images are resized to multiples of 28 pixels. If this still occurs, check that `src/prescription.py` has `_PATCH = 28` set.

### `Could not parse structured data from the model response`
The model returned output that could not be parsed as JSON. Check the console for `[json]` diagnostic lines showing exactly what was received. Usually caused by the response being truncated — try increasing `MAX_TOKENS` in `src/config.py`.

### History sidebar is empty
Records are only saved after a successful prescription analysis (image or PDF upload). Text-only conversations are not saved to the database.

### Port already in use
Gradio will automatically try the next available port and print the URL in the terminal.

### Slow first response
Normal — the model is warming up its KV cache. Subsequent responses in the same session are faster.
