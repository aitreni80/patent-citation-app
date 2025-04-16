import streamlit as st
import pdfplumber
from docx import Document
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_claims(text):
    claims = re.findall(r"(?i)claim\s*\d+.*?(?=claim\s*\d+|\Z)", text, re.DOTALL)
    return claims if claims else [text]  # fallback if no regex match

def extract_paragraphs(text):
    tagged_paragraphs = re.findall(r"(\[\d{4}\].*?)(?=\[\d{4}\]|$)", text, re.DOTALL)
    if tagged_paragraphs:
        return [para.strip() for para in tagged_paragraphs]
    fallback_paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in fallback_paragraphs if len(p.strip()) > 50]

def rank_relevant_paragraphs(claim, source_paragraphs, top_k=3):
    vectorizer = TfidfVectorizer().fit_transform([claim] + source_paragraphs)
    cosine_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    top_indices = cosine_matrix[0].argsort()[-top_k:][::-1]
    return [source_paragraphs[i] for i in top_indices]

# Streamlit UI
st.title("Patent Claim Auto-Citation Generator")

claims_file = st.file_uploader("Upload Claims Document (PDF/DOCX)", type=["pdf", "docx"])
citation_file = st.file_uploader("Upload Reference Document (PDF/DOCX)", type=["pdf", "docx"])

if claims_file and citation_file:
    # Extract claim text
    claim_text = extract_text_from_pdf(claims_file) if claims_file.type == "application/pdf" else extract_text_from_docx(claims_file)
    
    # Extract reference/citation text
    citation_text = extract_text_from_pdf(citation_file) if citation_file.type == "application/pdf" else extract_text_from_docx(citation_file)

    claims = extract_claims(claim_text)
    citation_paragraphs = extract_paragraphs(citation_text)

    st.success("Files uploaded and processed.")

    results = {}
    for i, claim in enumerate(claims, 1):
        best_matches = rank_relevant_paragraphs(claim, citation_paragraphs)
        results[f"Claim {i}"] = best_matches

    for claim, citations in results.items():
        st.markdown(f"### {claim}")
        for i, c in enumerate(citations, 1):
            st.markdown(f"> **[{i}]** {c}")

    st.download_button("Download Citation Mapping",
        "\n\n".join(f"{k}:\n" + "\n".join([f"[{i+1}] {s}" for i, s in enumerate(v)])
                      for k, v in results.items()),
        file_name="citation_mapping.txt"
    )
