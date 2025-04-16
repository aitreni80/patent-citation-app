# Patent Claim Auto-Citation Generator

This Streamlit web app matches claims in a patent document with relevant paragraphs from a supporting reference document.

## Features
- Upload claims and citation documents (PDF or DOCX)
- Automatically match claims to relevant paragraphs
- Detect paragraph tags (e.g., [0012]) or fallback to spacing
- Download citation mappings

## To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
