"""
Indian Medical Prescription Analyzer — entry point.
Run:  python index.py

Layout:
  src/config.py        paths & constants
  src/db.py            SQLite schema + CRUD
  src/llm.py           model loading (Gemma + Qwen2.5-VL)
  src/prescription.py  image helpers, JSON→Markdown
  src/ui.py            Gradio interface + chat callback
  prompts/
    default_prompt.txt vision model system prompt (edit freely)
    prescription_analysis.md  field reference & documentation
"""

from src.ui import prescripto, SIDEBAR_CSS, SETUP_JS

if __name__ == "__main__":
    prescripto.launch(css=SIDEBAR_CSS, js=SETUP_JS, share=False)
