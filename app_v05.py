# text_similarity_app.py
# ----------------------------------------
# Multi-Method Text Similarity App
# ----------------------------------------
# Text Similarity & Plagiarism Detector v0.4
# Developed by Al Yazdani, March 2026

import streamlit as st
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional embedding model
USE_EMBEDDINGS = True
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
except:
    USE_EMBEDDINGS = False


# ----------------------------------------
# Utility Functions
# ----------------------------------------

def levenshtein_ratio(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def ngram_jaccard(text1, text2, n=3):
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return set(zip(*[tokens[i:] for i in range(n)]))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))

    return intersection / union if union != 0 else 0


def tfidf_cosine(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


def embedding_similarity(text1, text2):
    if not USE_EMBEDDINGS:
        return None
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.set_page_config(page_title="Text Similarity Lab", layout="wide")

st.title("🧪 Multi-Method Text Similarity Analyzer")
st.subheader("Developed by: :blue[Al Yazdani]",divider=True)

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("**Text A**", "A Banach space is a vector space which is a complete metric space.",height=200)

with col2:
    text2 = st.text_area("**Text B**", "A Hilbert space is a Banach space that is endowed with an inner product.",height=200)

st.markdown("---")

if st.button("Compute Similarity"):

    if not text1 or not text2:
        st.warning("Please enter both texts.")
    else:
        results = {}

        # Compute metrics
        results["Levenshtein Ratio"] = levenshtein_ratio(text1, text2)
        results["Jaccard (words)"] = jaccard_similarity(text1, text2)
        results["Jaccard (3-grams)"] = ngram_jaccard(text1, text2, n=3)
        results["TF-IDF Cosine"] = tfidf_cosine(text1, text2)

        emb_score = embedding_similarity(text1, text2)
        if emb_score is not None:
            results["Embedding Cosine"] = emb_score
        else:
            results["Embedding Cosine"] = "Unavailable"

        # Convert to DataFrame
        df = pd.DataFrame(
            [(k, v) for k, v in results.items()],
            columns=["Method", "Score"]
        )

        # Normalize formatting
        df["Score"] = df["Score"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) else x
        )

        st.subheader("📊 Similarity Scores")
        st.dataframe(df, use_container_width=True)

        # Interpretation
        st.subheader("🧠 Interpretation Guide")

        st.markdown("""
- **0.9 – 1.0** → Nearly identical  
- **0.7 – 0.9** → Strong similarity  
- **0.4 – 0.7** → Moderate similarity  
- **< 0.4** → Weak similarity  
        """)

        # Quick insights
        st.subheader("🔍 Observations")

        if isinstance(results["TF-IDF Cosine"], float):
            if results["TF-IDF Cosine"] > 0.8 and (emb_score is None or emb_score < 0.7):
                st.info("High lexical overlap but weaker semantic similarity → possible keyword matching.")

            if emb_score is not None and emb_score > 0.85 and results["TF-IDF Cosine"] < 0.6:
                st.info("Low lexical overlap but high semantic similarity → likely paraphrasing.")


# ----------------------------------------
# Sidebar Options
# ----------------------------------------

st.sidebar.header("⚙️ Settings")

ngram_size = st.sidebar.slider("N-gram size", 2, 5, 3)

st.sidebar.markdown("""
### Notes
- Embeddings may be disabled if model cannot load
- TF-IDF works fully offline
- This app is extensible for:
  - BM25
  - Cross-encoders
  - LLM scoring
""")