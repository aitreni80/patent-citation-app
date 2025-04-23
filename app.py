import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Patent Claim Matcher", layout="wide")
st.title("Patent Claim Feature Matcher")

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_paragraphs_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def extract_features(claim_text):
    # Very simple feature splitter: each line or semicolon could be a feature
    features = [f.strip("•; \n") for f in claim_text.splitlines() if f.strip()]
    return features

def match_features_to_reference(features, reference_paragraphs):
    ref_embeddings = model.encode(reference_paragraphs, convert_to_tensor=True)
    results = []
    for feature in features:
        feature_embedding = model.encode(feature, convert_to_tensor=True)
        cosine_scores = util.cos_sim(feature_embedding, ref_embeddings)[0]
        best_match_idx = int(cosine_scores.argmax())
        best_score = float(cosine_scores[best_match_idx])
        results.append((feature, reference_paragraphs[best_match_idx], best_score))
    return results

# --- Streamlit UI ---
claim_text = st.text_area("Paste a single claim below:", height=250)

ref_file = st.file_uploader("Upload reference PDF", type=["pdf"])

if st.button("Match Features") and claim_text and ref_file:
    with st.spinner("Processing..."):
        reference_paragraphs = extract_paragraphs_from_pdf(ref_file)
        features = extract_features(claim_text)
        matches = match_features_to_reference(features, reference_paragraphs)

    st.subheader("Matched Features with Citations")
    for idx, (feature, para, score) in enumerate(matches, start=1):
        st.markdown(f"**Feature {idx}:** {feature}")
        st.markdown(f"*Matched Paragraph (Score: {score:.2f}):*")
        st.code(para)
        st.markdown("---")
