import streamlit as st
import fitz  # PyMuPDF
import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
from nltk.corpus import stopwords

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

def extract_key_phrases(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    return [word for word, tag in tagged if tag.startswith("NN") and word.lower() not in stop_words and word.isalnum()]

def match_claims_to_reference(claims, ref_paras):
    matches = []
    for claim in claims:
        claim_keywords = set(extract_key_phrases(claim))
        para_scores = []
        for para in ref_paras:
            para_keywords = set(extract_key_phrases(para))
            common = claim_keywords.intersection(para_keywords)
            score = len(common) / len(claim_keywords) if claim_keywords else 0
            para_scores.append((para, score))
        top_matches = sorted(para_scores, key=lambda x: x[1], reverse=True)[:3]
        matches.append([m for m in top_matches if m[1] > 0])
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
