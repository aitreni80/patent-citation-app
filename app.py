import streamlit as st
import pdfplumber
import spacy
import re
from typing import List, Dict

# Use a lightweight blank English model to avoid needing downloads
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")  # add sentence boundary detection

st.set_page_config(layout="wide")
st.title("Patent Claim Matcher")

def extract_text_from_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def extract_claims(text: str) -> List[str]:
    claim_starts = [m.start() for m in re.finditer(r'\b\d+\.', text)]
    claim_starts.append(len(text))
    claims = []
    for i in range(len(claim_starts) - 1):
        claim = text[claim_starts[i]:claim_starts[i + 1]].strip()
        if len(claim.split()) > 3:
            claims.append(claim)
    return claims

def extract_features(claim_text: str) -> List[str]:
    return [line.strip('• ').strip() for line in claim_text.split('\n') if line.strip().startswith('•')]

def extract_key_phrases(text: str) -> List[str]:
    doc = nlp(text.lower())
    return [token.text for token in doc if token.is_alpha and not token.is_stop]

def match_features_to_reference(claim: str, reference_paragraphs: List[str]) -> List[Dict]:
    features = extract_features(claim)
    matches = []
    for feature in features:
        keywords = set(extract_key_phrases(feature))
        best_para = ''
        best_score = 0
        best_index = -1
        for i, para in enumerate(reference_paragraphs):
            para_keywords = set(extract_key_phrases(para))
            score = len(keywords & para_keywords) / (len(keywords) + 1e-5)
            if score > best_score:
                best_score = score
                best_para = para
                best_index = i
        if best_score > 0.2:
            matches.append({
                'feature': feature,
                'para': best_para,
                'score': round(best_score, 2),
                'index': best_index
            })
    return matches

uploaded_claims = st.file_uploader("Upload Claims PDF", type=["pdf"])
uploaded_reference = st.file_uploader("Upload Reference PDF", type=["pdf"])

if uploaded_claims and uploaded_reference:
    claim_text = extract_text_from_pdf(uploaded_claims)
    reference_text = extract_text_from_pdf(uploaded_reference)
    claims = extract_claims(claim_text)
    ref_paragraphs = split_paragraphs(reference_text)

    st.subheader("Matched Claims")
    for idx, claim in enumerate(claims, 1):
        st.markdown(f"**Claim {idx}:**")
        matched_features = match_features_to_reference(claim, ref_paragraphs)
        for match in matched_features:
            st.markdown(f"- `{match['feature']}`")
            st.markdown(f"  - **D1 para. [{match['index']}], score={match['score']}**: {match['para']}")
