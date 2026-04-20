import os

# Project root is one level above this file (src/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DB_PATH    = os.path.join(BASE_DIR, "prescriptions.db")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

TEXT_MODEL_PATH    = os.path.join(MODELS_DIR, "google_gemma-4-E4B-it-Q4_K_M.gguf")
VISION_MODEL_PATH  = os.path.join(MODELS_DIR, "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
VISION_MMPROJ_PATH = os.path.join(MODELS_DIR, "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")

N_GPU_LAYERS = -1
N_CTX_TEXT   = 8192
N_CTX_VISION = 16384   # ~1600 img tokens (1120px) + 2k prompt + 4k output
MAX_TOKENS   = 4096
TEMPERATURE  = 0.1

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

MODE_MAP = {
    "Patient Wise": "patient",
    "Doctor Wise":  "doctor",
    "General":      "general",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
