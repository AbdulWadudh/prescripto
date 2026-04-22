import os
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from .config import (
    TEXT_MODEL_PATH, VISION_MODEL_PATH, VISION_MMPROJ_PATH,
    N_GPU_LAYERS, N_CTX_TEXT, N_CTX_VISION,
)

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
        verbose=False,
    )
    # Probe mtmd lazy init — needs a >=2x2 image
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
    print("[load] text model ready (vision disabled)")
