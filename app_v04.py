# Text Similarity & Plagiarism Detector v0.4
# Developed by Al Yazdani, March 2026
# ─────────────────────────────────────────────────────────────────────────────
# SETUP (one-time, then fully offline):
#   pip install streamlit scikit-learn nltk spacy pandas plotly
#   python -m spacy download en_core_web_lg
# ─────────────────────────────────────────────────────────────────────────────

import re
import streamlit as st
from difflib import SequenceMatcher
from io import StringIO

import pandas as pd
import plotly.graph_objects as go

import nltk
nltk.download("punkt",       quiet=True)
nltk.download("punkt_tab",   quiet=True)
nltk.download("stopwords",   quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util    import ngrams
from nltk.corpus  import stopwords as nltk_stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

import spacy   # offline model — no internet required after install

# ─────────────────────────────────────────────────────────────────────────────
# Load spaCy model (en_core_web_lg ships 300-dim GloVe vectors, 100 % offline)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        st.error(
            "⚠️ spaCy model not found.  "
            "Run once in your terminal:  \n"
            "```\npython -m spacy download en_core_web_lg\n```"
        )
        st.stop()

nlp = load_spacy_model()

# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing helper
# ─────────────────────────────────────────────────────────────────────────────
_STOP_WORDS = set(nltk_stopwords.words("english"))

def preprocess(text: str, remove_stops: bool = False) -> str:
    """Lowercase and optionally strip stopwords / punctuation."""
    text = text.lower().strip()
    if remove_stops:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and t not in _STOP_WORDS]
        text = " ".join(tokens)
    return text

# ─────────────────────────────────────────────────────────────────────────────
# Similarity metrics
# ─────────────────────────────────────────────────────────────────────────────
def char_similarity(t1: str, t2: str) -> float:
    """Character-level edit distance ratio (SequenceMatcher)."""
    return SequenceMatcher(None, t1, t2).ratio()

def jaccard_similarity(t1: str, t2: str) -> float:
    """Word-level Jaccard index (intersection / union)."""
    s1 = set(word_tokenize(t1.lower()))
    s2 = set(word_tokenize(t2.lower()))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0

def tfidf_similarity(t1: str, t2: str) -> float:
    """TF-IDF weighted cosine similarity (word + character n-grams)."""
    # Combine word-level and char-level TF-IDF for better coverage
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    try:
        w = cosine_similarity(vec_word.fit_transform([t1, t2])[0],
                              vec_word.fit_transform([t1, t2])[1])[0][0]
        c = cosine_similarity(vec_char.fit_transform([t1, t2])[0],
                              vec_char.fit_transform([t1, t2])[1])[0][0]
        return 0.6 * w + 0.4 * c          # word patterns matter more
    except Exception:
        return 0.0

def ngram_overlap(t1: str, t2: str, n: int = 2) -> float:
    """
    Overlap coefficient for word n-grams:
        |A ∩ B| / min(|A|, |B|)
    Robust to length differences; good at catching reordered passages.
    """
    toks1 = word_tokenize(t1.lower())
    toks2 = word_tokenize(t2.lower())
    ng1 = set(ngrams(toks1, n))
    ng2 = set(ngrams(toks2, n))
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / min(len(ng1), len(ng2))

def spacy_semantic_similarity(t1: str, t2: str) -> float:
    """
    Semantic similarity via spaCy's en_core_web_lg 300-dim GloVe vectors.
    Works 100 % offline — no API or internet access required.
    """
    doc1 = nlp(t1)
    doc2 = nlp(t2)
    if not doc1.has_vector or not doc2.has_vector:
        st.warning("One or both texts lack vector representations (possible OOV issue).")
        return 0.0
    return float(doc1.similarity(doc2))

# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level matching
# ─────────────────────────────────────────────────────────────────────────────
def top_sentence_pairs(t1: str, t2: str, top_n: int = 5):
    """
    Cross-compare every sentence pair; return top_n by TF-IDF cosine score.
    Useful for spotting locally plagiarised passages.
    """
    sents1 = sent_tokenize(t1)
    sents2 = sent_tokenize(t2)
    pairs  = []
    for s1 in sents1:
        for s2 in sents2:
            if len(s1.split()) < 3 or len(s2.split()) < 3:
                continue                 # skip trivially short sentences
            vec = TfidfVectorizer().fit_transform([s1, s2])
            score = cosine_similarity(vec[0], vec[1])[0][0]
            pairs.append((s1.strip(), s2.strip(), score))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]

# ─────────────────────────────────────────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────────────────────────────────────────
def interpret(score: float) -> str:
    if score >= 0.85: return "🔴 Very high similarity (near-duplicate)"
    if score >= 0.70: return "🟠 Strong similarity"
    if score >= 0.50: return "🟡 Moderate similarity"
    if score >= 0.30: return "🟢 Weak similarity"
    return               "🟢 Very low similarity"

def plagiarism_verdict(score: float) -> str:
    if score >= 0.85: return "🚨 **Likely plagiarism** — texts are near-identical."
    if score >= 0.70: return "⚠️ **High risk** — substantial shared content."
    if score >= 0.50: return "⚠️ **Moderate risk** — notable overlap; review advised."
    if score >= 0.30: return "✅ **Low risk** — limited overlap, likely coincidental."
    return               "✅ **Minimal risk** — texts appear independently written."

# ─────────────────────────────────────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────────────────────────────────────
def radar_chart(scores: dict) -> go.Figure:
    labels = list(scores.keys())
    vals   = list(scores.values())
    vals  += [vals[0]]         # close the polygon
    labels_plot = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=labels_plot,
        fill="toself", fillcolor="rgba(99,110,250,0.25)",
        line=dict(color="rgba(99,110,250,0.9)", width=2),
        marker=dict(size=6)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20),
        height=350
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Character-level diff highlighter
# ─────────────────────────────────────────────────────────────────────────────
def highlight_diff(t1: str, t2: str) -> tuple[str, str]:
    """Return HTML strings with matching spans highlighted in green."""
    matcher = SequenceMatcher(None, t1, t2)
    h1, h2 = [], []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        seg_a = t1[i1:i2]
        seg_b = t2[j1:j2]
        if op == "equal":
            style = 'background:#d4edda; border-radius:3px;'
            h1.append(f'<span style="{style}">{seg_a}</span>')
            h2.append(f'<span style="{style}">{seg_b}</span>')
        else:
            h1.append(seg_a)
            h2.append(seg_b)
    return "".join(h1), "".join(h2)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Text Similarity & Plagiarism Detector", layout="wide")

st.title("📄 Text Similarity & Plagiarism Detector")
st.header("Developed by: :blue[Al Yazdani]", divider=True)

st.markdown(
    "Comparing two texts using **six complementary metrics**: "
    "character-level, Jaccard, TF-IDF (word + char n-grams), "
    "bigram overlap, trigram overlap, and **offline semantic (GloVe/spaCy)** similarity."
)

with st.expander("📋 Instructions", expanded=False):
    st.markdown(
        """
1. Enter two texts below (**max 1 000 words each**).
2. Toggle preprocessing options as needed.
3. Adjust the **six metric weights** (they are auto-normalised).
4. Click **Run Similarity Analysis**.
5. Review scores, the radar chart, the sentence-pair table, and the diff view.
6. Download a full CSV report.
"""
    )

# ── Text inputs ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area("**Text A**","In recent years, machine learning has emerged as a fundamental tool in quantitative research, enabling the extraction of complex patterns from large-scale datasets. Supervised learning techniques, in particular, have demonstrated strong performance in predictive modeling tasks, where labeled data is available. However, the effectiveness of these models depends critically on the quality of the input features and the assumptions underlying the training process. As a result, careful data preprocessing, feature engineering, and model validation are essential components of any robust machine learning pipeline.",height=200)

with col2:
    text2 = st.text_area("**Text B**", "Over the past decade, machine learning has become an essential methodology in quantitative analysis, allowing researchers to uncover intricate structures within vast datasets. Among various approaches, supervised learning methods have proven especially effective for prediction problems that rely on labeled observations. Nevertheless, model performance is highly sensitive to the choice of input variables and the assumptions made during training. Consequently, rigorous data preparation, thoughtful feature construction, and thorough validation procedures play a crucial role in building reliable machine learning systems.",height=200) 


def word_count(text: str) -> int:
    return len(text.split())

wc1, wc2 = word_count(text1), word_count(text2)
col1.caption(f"Text A: {wc1} / 1 000 words")
col2.caption(f"Text B: {wc2} / 1 000 words")

if wc1 > 1000 or wc2 > 1000:
    st.error("Each text must be 1 000 words or fewer.")
    st.stop()

# ── Pre-processing options ───────────────────────────────────────────────────
st.subheader("🔧 Pre-processing Options")
remove_stops = st.checkbox(
    "Remove stopwords before analysis",
    value=False,
    help="Strips common words (the, is, a…) before computing lexical metrics. "
         "Can improve precision but may reduce recall for stylistic matching."
)

# ── Weight sliders ───────────────────────────────────────────────────────────
st.subheader("⚖️ Metric Weights")
st.caption("Weights are auto-normalised to 1.0 — adjust freely.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    w_char    = st.slider("Character similarity",    0.0, 1.0, 0.10, 0.05)
    w_jaccard = st.slider("Jaccard similarity",      0.0, 1.0, 0.10, 0.05)
with col_b:
    w_tfidf   = st.slider("TF-IDF cosine",           0.0, 1.0, 0.25, 0.05)
    w_bigram  = st.slider("Bigram overlap",          0.0, 1.0, 0.15, 0.05)
with col_c:
    w_trigram = st.slider("Trigram overlap",         0.0, 1.0, 0.10, 0.05)
    w_spacy   = st.slider("Semantic (spaCy/GloVe)",  0.0, 1.0, 0.30, 0.05)

weight_sum = w_char + w_jaccard + w_tfidf + w_bigram + w_trigram + w_spacy
if weight_sum == 0:
    st.error("All weights are 0 — please set at least one weight above 0.")
    st.stop()

# Normalise
def norm(w): return w / weight_sum
nw = {k: norm(v) for k, v in {
    "char": w_char, "jaccard": w_jaccard, "tfidf": w_tfidf,
    "bigram": w_bigram, "trigram": w_trigram, "spacy": w_spacy
}.items()}

# ── Run analysis ─────────────────────────────────────────────────────────────
if st.button("🔍 Run Similarity Analysis", type="primary"):
    if not text1.strip() or not text2.strip():
        st.error("Please enter text in both boxes.")
        st.stop()

    p1 = preprocess(text1, remove_stops=remove_stops)
    p2 = preprocess(text2, remove_stops=remove_stops)

    with st.spinner("Computing similarity scores…"):
        char_s    = char_similarity(text1, text2)        # raw text (char diff)
        jac_s     = jaccard_similarity(p1, p2)
        tfidf_s   = tfidf_similarity(p1, p2)
        bigram_s  = ngram_overlap(p1, p2, n=2)
        trigram_s = ngram_overlap(p1, p2, n=3)
        spacy_s   = spacy_semantic_similarity(text1, text2)  # always raw text

        weighted = (
            nw["char"]    * char_s    +
            nw["jaccard"] * jac_s     +
            nw["tfidf"]   * tfidf_s   +
            nw["bigram"]  * bigram_s  +
            nw["trigram"] * trigram_s +
            nw["spacy"]   * spacy_s
        )

    scores_dict = {
        "Character":  char_s,
        "Jaccard":    jac_s,
        "TF-IDF":     tfidf_s,
        "Bigram":     bigram_s,
        "Trigram":    trigram_s,
        "Semantic\n(spaCy)": spacy_s,
    }

    # ── Individual score cards ────────────────────────────────────────────────
    st.subheader("📊 Similarity Scores")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    def score_card(col, label, val, tooltip=""):
        col.metric(label, f"{val:.3f}", help=tooltip)
        col.caption(interpret(val))

    score_card(c1, "Character similarity",   char_s,
               "Edit-distance ratio on raw characters. Sensitive to paraphrasing.")
    score_card(c2, "Jaccard similarity",      jac_s,
               "Word-set intersection / union. Ignores word order.")
    score_card(c3, "TF-IDF cosine",           tfidf_s,
               "Word + character n-gram TF-IDF cosine. Balances frequency and rarity.")
    score_card(c4, "Bigram overlap",          bigram_s,
               "Overlap coefficient on consecutive word pairs. Catches phrase reuse.")
    score_card(c5, "Trigram overlap",         trigram_s,
               "Overlap coefficient on word triples. High precision phrase matching.")
    score_card(c6, "Semantic (spaCy/GloVe)",  spacy_s,
               "300-dim GloVe vectors via spaCy en_core_web_lg — fully offline.")

    st.divider()

    # ── Weighted overall ──────────────────────────────────────────────────────
    st.subheader("⭐ Weighted Overall Similarity")
    ov_col, verd_col = st.columns([1, 2])
    ov_col.metric("Unified score (auto-normalised weights)", f"{weighted:.3f}")
    ov_col.caption(interpret(weighted))
    verd_col.markdown("### Plagiarism Verdict")
    verd_col.markdown(plagiarism_verdict(weighted))

    st.divider()

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("🕸️ Metric Radar Chart")
    st.plotly_chart(radar_chart(scores_dict), use_container_width=True)

    st.divider()

    # ── Sentence-level matching ───────────────────────────────────────────────
    st.subheader("🔎 Top Matching Sentence Pairs")
    st.caption("Highest-similarity cross-sentence pairs — useful for spotting locally reused passages.")
    pairs = top_sentence_pairs(text1, text2, top_n=5)
    if pairs:
        df_pairs = pd.DataFrame(pairs, columns=["Sentence from A", "Sentence from B", "TF-IDF Score"])
        df_pairs["TF-IDF Score"] = df_pairs["TF-IDF Score"].map("{:.3f}".format)
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough sentence content to compare at the sentence level.")

    st.divider()

    # ── Character-level diff ──────────────────────────────────────────────────
    with st.expander("🖍️ Shared-passage Highlighter (character-level diff)", expanded=False):
        st.caption("Green = matching spans between the two texts.")
        h1, h2 = highlight_diff(text1, text2)
        dc1, dc2 = st.columns(2)
        dc1.markdown(f"**Text A**<br><div style='font-size:0.9em;line-height:1.6'>{h1}</div>",
                     unsafe_allow_html=True)
        dc2.markdown(f"**Text B**<br><div style='font-size:0.9em;line-height:1.6'>{h2}</div>",
                     unsafe_allow_html=True)

    st.divider()

    # ── Narrative summary ─────────────────────────────────────────────────────
    st.subheader("🧠 Summary")
    lexical_avg  = (char_s + jac_s) / 2
    ngram_avg    = (bigram_s + trigram_s) / 2
    st.markdown(
        f"""
| Dimension | Score | Assessment |
|---|---|---|
| **Lexical** (character + Jaccard)  | {lexical_avg:.3f} | {interpret(lexical_avg)}  |
| **Statistical** (TF-IDF)           | {tfidf_s:.3f}     | {interpret(tfidf_s)}      |
| **Phrase overlap** (bi- + trigram) | {ngram_avg:.3f}   | {interpret(ngram_avg)}    |
| **Semantic** (spaCy GloVe)         | {spacy_s:.3f}     | {interpret(spacy_s)}      |
| **Overall** (weighted)             | {weighted:.3f}    | {interpret(weighted)}     |
"""
    )

    st.divider()

    # ── CSV download ──────────────────────────────────────────────────────────
    results = {
        "Metric": [
            "Character similarity",
            "Jaccard similarity",
            "TF-IDF cosine (word+char ngrams)",
            "Bigram overlap",
            "Trigram overlap",
            "Semantic similarity (spaCy GloVe)",
            "Weighted overall similarity",
        ],
        "Score": [char_s, jac_s, tfidf_s, bigram_s, trigram_s, spacy_s, weighted],
        "Interpretation": [
            interpret(char_s), interpret(jac_s), interpret(tfidf_s),
            interpret(bigram_s), interpret(trigram_s), interpret(spacy_s),
            interpret(weighted),
        ],
        "Normalised weight": [
            nw["char"], nw["jaccard"], nw["tfidf"],
            nw["bigram"], nw["trigram"], nw["spacy"], "—",
        ],
    }
    df_results = pd.DataFrame(results)

    # Append the raw texts as metadata rows
    meta = pd.DataFrame({
        "Metric": ["", "Text A", "Text B"],
        "Score": ["", "", ""],
        "Interpretation": ["", "", ""],
        "Normalised weight": ["", text1, text2],
    })
    df_export = pd.concat([df_results, meta], ignore_index=True)

    csv_buffer = StringIO()
    df_export.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Full Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name="text_similarity_results.csv",
        mime="text/csv",
    )
