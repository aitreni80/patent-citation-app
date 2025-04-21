# app.py
import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import os

nltk.download("punkt")
nltk.download("stopwords")

st.title("Patent Claim Citation Matcher")

# ---------------------- Helper Functions ----------------------
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def split_claims(text):
    claims = re.split(r"(?<=\n)\s*(\d+\.)", text)
    if len(claims) < 2:
        # fallback to line spacing if no numbers
        return [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]
    grouped = []
    i = 1
    while i < len(claims):
        number = claims[i]
        content = claims[i + 1] if i + 1 < len(claims) else ""
        grouped.append(content.strip())
        i += 2
    return grouped

def split_paragraphs(text):
    paras = re.split(r"\n\s*\n|\n(?=\[\d{4}\])", text)
    return [p.strip() for p in paras if len(p.strip()) > 30]

def extract_key_phrases(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
    return words

def match_features_to_reference(claim, reference_paragraphs, top_n=3):
    features = [f.strip("-• ") for f in claim.split("\n") if len(f.strip()) > 10]
    matches = []
    for feature in features:
        keywords = extract_key_phrases(feature)
        para_scores = []
        for i, para in enumerate(reference_paragraphs):
            ref_keywords = extract_key_phrases(para)
            overlap = set(keywords) & set(ref_keywords)
            score = len(overlap) / (len(set(keywords)) + 1e-5)
            if score > 0:
                para_scores.append((i, para, score))
        para_scores.sort(key=lambda x: x[2], reverse=True)
        top_matches = para_scores[:top_n]
        if top_matches:
            matches.append((feature, top_matches))
    return matches

# ---------------------- UI ----------------------
claims_pdf = st.file_uploader("Upload Claims PDF", type="pdf")
ref_pdf = st.file_uploader("Upload Reference PDF (e.g., D1)", type="pdf")

document_label = st.text_input("Enter reference document label (e.g., D1)", "D1")

if st.button("Match Claims") and claims_pdf and ref_pdf:
    with st.spinner("Extracting and analyzing..."):
        claims_text = extract_text_from_pdf(claims_pdf)
        reference_text = extract_text_from_pdf(ref_pdf)

        claims = split_claims(claims_text)
        ref_paragraphs = split_paragraphs(reference_text)

        for idx, claim in enumerate(claims, start=1):
            st.subheader(f"Claim {idx}:")
            st.markdown(f"> {claim}")
            matched_features = match_features_to_reference(claim, ref_paragraphs)
            for feature, top_matches in matched_features:
                st.markdown(f"**• {feature}**")
                for match_idx, para, score in top_matches:
                    st.markdown(f"*{document_label} para. [{match_idx}]:* {para[:500]}...  ")
            st.markdown("---")
