# Text Similarity & Plagiarism Detector 
# Developed by Al Yazdani, March 2026
import streamlit as st
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

# ----------------------------
# Load embedding model once
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# define Similarity functions
# ----------------------------
def char_similarity(t1, t2):
    return SequenceMatcher(None, t1, t2).ratio()

def jaccard_similarity(t1, t2):
    s1 = set(word_tokenize(t1.lower()))
    s2 = set(word_tokenize(t2.lower()))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0

def tfidf_similarity(t1, t2):
    tfidf = TfidfVectorizer().fit_transform([t1, t2])
    return cosine_similarity(tfidf[0], tfidf[1])[0][0]

def embedding_similarity(t1, t2):
    emb = model.encode([t1, t2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]



# ----------------------------
# Interpretation helper
# ----------------------------
def interpret(score):
    if score >= 0.85:
        return "Very high similarity (near-duplicate)"
    elif score >= 0.70:
        return "Strong similarity"
    elif score >= 0.50:
        return "Moderate similarity"
    elif score >= 0.30:
        return "Weak similarity"
    else:
        return "Very low similarity"

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Text Similarity/Plagiarism Detector")
st.subheader("Developed by: :blue[Al Yazdani]",divider=True)
st.markdown(
    """
Compare two texts using **lexical and semantic similarity metrics**.
Each input is limited to **500 words**.
"""
)

# Text inputs
text1 = st.text_area("**Text A**","In recent years, machine learning has emerged as a fundamental tool in quantitative research, enabling the extraction of complex patterns from large-scale datasets. Supervised learning techniques, in particular, have demonstrated strong performance in predictive modeling tasks, where labeled data is available. However, the effectiveness of these models depends critically on the quality of the input features and the assumptions underlying the training process. As a result, careful data preprocessing, feature engineering, and model validation are essential components of any robust machine learning pipeline.",height=200)
text2 = st.text_area("**Text B**", " Over the past decade, machine learning has become an essential methodology in quantitative analysis, allowing researchers to uncover intricate structures within vast datasets. Among various approaches, supervised learning methods have proven especially effective for prediction problems that rely on labeled observations. Nevertheless, model performance is highly sensitive to the choice of input variables and the assumptions made during training. Consequently, rigorous data preparation, thoughtful feature construction, and thorough validation procedures play a crucial role in building reliable machine learning systems.",height=200) 

# Word count check
def word_count(text):
    return len(text.split())

wc1, wc2 = word_count(text1), word_count(text2)

st.caption(f"Text A: {wc1}/500 words")
st.caption(f"Text B: {wc2}/500 words")

if wc1 > 500 or wc2 > 500:
    st.error("Each text must be 500 words or fewer.")
    st.stop()

# Weight sliders
st.subheader("⚖️ Similarity Weights")

w_char = st.slider("Character similarity weight", 0.0, 1.0, 0.15)
w_jaccard = st.slider("Jaccard similarity weight", 0.0, 1.0, 0.15)
w_tfidf = st.slider("TF-IDF similarity weight", 0.0, 1.0, 0.30)
w_embed = st.slider("Embedding similarity weight", 0.0, 1.0, 0.40)

weight_sum = w_char + w_jaccard + w_tfidf + w_embed
if abs(weight_sum - 1.0) > 1e-6:
    st.warning(f"Weights sum to {weight_sum:.2f}. Consider normalizing to 1.0.")

# Run analysis
if st.button("🔍 Run Similarity Analysis"):
    if not text1.strip() or not text2.strip():
        st.error("Please enter text in both boxes.")
        st.stop()

    with st.spinner("Computing similarity scores..."):
        char = char_similarity(text1, text2)
        jac = jaccard_similarity(text1, text2)
        tfidf = tfidf_similarity(text1, text2)
        embed = embedding_similarity(text1, text2)

        weighted = (
            w_char * char +
            w_jaccard * jac +
            w_tfidf * tfidf +
            w_embed * embed
        )

    # Results
    st.subheader("📊 Similarity Scores")

    st.metric("Character similarity", f"{char:.3f}")
    st.write(interpret(char))

    st.metric("Jaccard similarity", f"{jac:.3f}")
    st.write(interpret(jac))

    st.metric("TF-IDF similarity", f"{tfidf:.3f}")
    st.write(interpret(tfidf))

    st.metric("Embedding (semantic) similarity", f"{embed:.3f}")
    st.write(interpret(embed))

    st.divider()

    st.subheader("⭐ Weighted Overall Similarity")
    st.metric("Unified similarity score", f"{weighted:.3f}")
    st.write(interpret(weighted))

    st.divider()

    st.subheader("🧠 Summary")
    st.markdown(
        f"""
- **Lexical similarity** (character & word overlap) is **{interpret((char + jac)/2).lower()}**
- **Statistical similarity** (TF-IDF) is **{interpret(tfidf).lower()}**
- **Semantic similarity** (embeddings) is **{interpret(embed).lower()}**
- **Overall**, the two texts are **{interpret(weighted).lower()}**
        """
    )


    
# add functionality to download into a csv the two texts, the scores and interpretations and weights used
    import pandas as pd
    from io import StringIO

    results = {
        "Metric": [
            "Character similarity",
            "Jaccard similarity",
            "TF-IDF similarity",
            "Embedding similarity",
            "Weighted overall similarity"
        ],
        "Score": [char, jac, tfidf, embed, weighted],
        "Interpretation": [
            interpret(char),
            interpret(jac),
            interpret(tfidf),
            interpret(embed),
            interpret(weighted)
        ],
        "Weight": [w_char, w_jaccard, w_tfidf, w_embed, "N/A"],
        "texts": [text1, text2, "", "", ""]
    }

    df = pd.DataFrame(results)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="text_similarity_results.csv",
        mime="text/csv"
    )