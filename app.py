import streamlit as st
import fitz  # PyMuPDF
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_paragraphs(text):
    paras = re.split(r"\n\s*\n|\[\d{4}\]", text)
    return [p.strip() for p in paras if len(p.strip()) > 20]

def extract_claims(text):
    claims = re.split(r"(?<=\n)\d+\.\s", text)
    return [c.strip() for c in claims if len(c.strip()) > 20]

def match_claims_to_reference(claims, ref_paras):
    vectorizer = TfidfVectorizer().fit(claims + ref_paras)
    claim_vectors = vectorizer.transform(claims)
    ref_vectors = vectorizer.transform(ref_paras)

    matches = []
    for i, c_vec in enumerate(claim_vectors):
        sims = cosine_similarity(c_vec, ref_vectors).flatten()
        top_indices = sims.argsort()[-3:][::-1]
        matched_paras = [(ref_paras[j], sims[j]) for j in top_indices if sims[j] > 0.1]
        matches.append(matched_paras)
    return matches

def format_output(claims, matches):
    output = ""
    for i, (claim, refs) in enumerate(zip(claims, matches)):
        output += f"\n**Claim {i+1}:**\n"
        output += claim + "\n"
        if refs:
            for idx, (ref, score) in enumerate(refs):
                output += f"\n> D1 para. (match {idx+1}, score={score:.2f}): {ref}\n"
        else:
            output += "\n> No relevant disclosure found in D1.\n"
    return output

st.title("Patent Claims Citation Matcher")

claims_file = st.file_uploader("Upload Claims Document (PDF)", type="pdf")
ref_file = st.file_uploader("Upload Reference Document (PDF)", type="pdf")

if claims_file and ref_file:
    with st.spinner("Extracting and processing text..."):
        claims_text = extract_text_from_pdf(claims_file)
        ref_text = extract_text_from_pdf(ref_file)

        claims = extract_claims(claims_text)
        ref_paras = split_into_paragraphs(ref_text)

        matches = match_claims_to_reference(claims, ref_paras)
        result = format_output(claims, matches)

    st.markdown(result)
