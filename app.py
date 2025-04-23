from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Split claim into features
def split_claim_into_features(claim_text):
    features = []
    # Naive split: by semicolon or 'and' with some preprocessing
    for part in claim_text.split(";"):
        part = part.strip()
        if len(part) > 30:  # Filter out short irrelevant chunks
            features.append(part)
    return features

# Step 2: Embed features and reference paragraphs
def embed_text_chunks(text_chunks):
    return model.encode(text_chunks, convert_to_tensor=True)

# Step 3: Match each feature to best paragraph(s)
def match_features_to_reference(features, reference_paragraphs, top_k=1, score_threshold=0.5):
    features_emb = embed_text_chunks(features)
    reference_emb = embed_text_chunks(reference_paragraphs)

    results = []
    for idx, feat_emb in enumerate(features_emb):
        scores = cosine_similarity([feat_emb.cpu().numpy()], reference_emb.cpu().numpy())[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        matches = []
        for i in top_indices:
            score = scores[i]
            if score >= score_threshold:
                matches.append({
                    "score": float(score),
                    "paragraph": reference_paragraphs[i]
                })
        results.append({
            "feature": features[idx],
            "matches": matches
        })
    return results

# Example usage
claim_text = """
An environmental impact management device comprising: an emission right acquiring unit to acquire emission right data indicating a right to emit a predetermined amount of chemical substance; a load acquiring unit to acquire manufacture load data indicating an amount of the chemical substance emitted by manufacture of an article; and a use fee calculating unit to calculate a use fee of the article on the basis of a price of the emission right data and the amount of the chemical substance emitted by manufacture of the article.
"""

reference_paragraphs = [  # Each para should be a separate clean paragraph
    "Processes for determining carbon dioxide emission impact. In one process, a carbon dioxide emission impact is determined for each stage of a multi-stage trip...",
    "[0006] Therefore, there is a need for new processes for understanding the carbon impact of a vehicle over its entire lifetime...",
    # Add more paragraphs here
]

features = split_claim_into_features(claim_text)
matches = match_features_to_reference(features, reference_paragraphs, top_k=1, score_threshold=0.4)

# Display results
for match in matches:
    print(f"\nFeature: {match['feature']}")
    for m in match["matches"]:
        print(f"  Matched Paragraph (Score: {m['score']:.2f}): {m['paragraph'][:200]}...")
