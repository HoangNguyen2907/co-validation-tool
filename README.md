# CO Validation Tool

Streamlit app for validating Packing List, Commercial Invoice, and CO PDF data.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run co-validator.py
```

If OCR is needed locally, install the Tesseract command-line tool too:

```bash
brew install tesseract
```