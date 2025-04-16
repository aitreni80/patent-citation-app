import streamlit as st
import pdfplumber
from docx import Document
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract claims from the claims document
def extract_claims(text):
    claims = re.findall(r"(?i)claim\s*\d+.*?(?=claim\s*\d+|\Z)", text, re.DOTALL)
    return claims if claims else [text]  # fallback if no regex match

# Extract paragraphs from the reference document
def extract_paragraphs(text):
    # First check for paragraphs with tags like [0012]
    tagged_paragraphs = re.findall(r"(\[\d{4}\].*?)(?=\[\d{4}\]|$)", text, re.DOTALL)
    if tagged_paragraphs:
        return [para.strip() for para in tagged_paragraphs]
    
    # If no tags found, fallback to splitting by blank lines
    fallback_paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in fallback_paragraphs if len(p.strip()) > 50]

# Rank reference paragraphs based on similarity to claim feature
def rank_relevant_paragraphs(claim_feature, reference_paragraphs, top_k=3):
    vectorizer = TfidfVectorizer().fit_transform([claim_feature] + reference_paragraphs)
    cosine_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    top_indices = cosine_matrix[0].argsort()[-top_k:][::-1]
    return [reference_paragraphs[i] for i in top_indices]

# Generate citation for claims with reference to the matching paragraphs
def generate_citations_for_claims(claims, reference_paragraphs):
    results = {}
    for i, claim in enumerate(claims, 1):
        claim_features = re.split(r'\n|•', claim)
        claim_results = []
        for feature in claim_features:
            feature = feature.strip()
            if not feature:
                continue
            best_matches = rank_relevant_paragraphs(feature, reference_paragraphs)
            claim_results.append((feature, best_matches))
        results[f"Claim {i}"] = claim_results
    return results

# Streamlit app structure
import streamlit as st

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

    citation_results = generate_citations_for_claims(claims, citation_paragraphs)

    # Display results
    for claim, features in citation_results.items():
        st.markdown(f"### {claim}")
        for feature, matches in features:
            st.markdown(f"> **Feature**: {feature}")
            for i, match in enumerate(matches, 1):
                st.markdown(f"> **[{i}]** {match}")

    st.download_button("Download Citation Mapping",
        "\n\n".join(f"{k}:\n" + "\n".join([f"[{i+1}] {s}" for i, s in enumerate(v)]) 
                      for k, v in citation_results.items()),
        file_name="citation_mapping.txt"
    )
