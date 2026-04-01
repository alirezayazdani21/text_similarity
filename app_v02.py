# ─────────────────────────────────────────────────────────────────────────────
# Text Similarity & Plagiarism Detector  v2.0
# Developed by Al Yazdani, March 2026
#
# NLI now uses CrossEncoder('cross-encoder/nli-deberta-v3-small') directly —
# no transformers.pipeline, no parsing ambiguity, significantly faster.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from sentence_transformers import SentenceTransformer, CrossEncoder

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Similarity & Plagiarism Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (light, editorial theme) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:         #f7f5f0;
    --surface:    #ffffff;
    --surface2:   #f0ede6;
    --border:     #ddd9d0;
    --border2:    #c8c3b8;
    --accent:     #1a6b5a;
    --accent-lt:  #e8f4f1;
    --accent2:    #0f4d40;
    --gold:       #b8860b;
    --gold-lt:    #fdf8ec;
    --red:        #c0392b;
    --red-lt:     #fdf0ee;
    --amber:      #d97706;
    --amber-lt:   #fffbeb;
    --text:       #1a1916;
    --text2:      #3d3a34;
    --muted:      #7a7670;
    --muted2:     #a09c95;
    --radius:     8px;
    --shadow:     0 1px 4px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.05);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
section.main > div { padding-top: 0 !important; }

/* ── Hero ── */
.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 0 0 var(--radius) var(--radius);
    padding: 2.2rem 2.8rem 2rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow);
}
.hero-eyebrow {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .55rem;
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 .45rem;
    line-height: 1.2;
}
.hero-sub {
    font-size: .9rem;
    color: var(--muted);
    margin: 0;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    margin-top: .9rem;
}
.pill {
    background: var(--accent-lt);
    border: 1px solid #b8d8d2;
    color: var(--accent2);
    border-radius: 20px;
    padding: .2rem .75rem;
    font-size: .73rem;
    font-weight: 600;
    letter-spacing: .03em;
}

/* ── Section headings ── */
.sec-eyebrow {
    font-size: .68rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: var(--accent); margin-bottom: .35rem;
}
.sec-title {
    font-family: 'Lora', serif;
    font-size: 1.25rem; font-weight: 600;
    color: var(--text); margin: 0 0 .2rem;
}
.sec-body {
    font-size: .88rem; color: var(--muted); margin: 0 0 1rem;
}
hr.div { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── Text label above textareas ── */
.txt-label {
    display: flex; align-items: center; gap: .5rem;
    font-weight: 600; font-size: .9rem; color: var(--text2);
    margin-bottom: .35rem;
}
.txt-label .badge {
    background: var(--accent); color: #fff;
    border-radius: 5px; padding: .1rem .5rem;
    font-size: .72rem; font-weight: 700;
}
.wc { font-size: .78rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; margin-top: .3rem; }
.wc.over { color: var(--red); font-weight: 600; }

/* ── Streamlit text area overrides ── */
textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: .92rem !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,.04) !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(26,107,90,.12) !important;
}

/* ── Primary run button ── */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    padding: .65rem 2rem !important;
    letter-spacing: .02em !important;
    box-shadow: 0 2px 8px rgba(26,107,90,.25) !important;
    transition: background .15s, transform .1s, box-shadow .15s !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--accent2) !important;
    box-shadow: 0 4px 14px rgba(26,107,90,.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
}

/* ── Overall score card ── */
.overall-card {
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    border: 1px solid;
    box-shadow: var(--shadow);
    margin: 1.5rem 0;
}
.overall-pct {
    font-family: 'Lora', serif;
    font-size: 3.4rem;
    font-weight: 600;
    margin: 0;
    line-height: 1;
}
.overall-verdict { font-size: 1rem; margin: .6rem 0 0; }

/* ── Metric cards ── */
.mc {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.mc::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: var(--radius) 0 0 var(--radius);
}
.mc-green::before  { background: var(--accent); }
.mc-amber::before  { background: var(--amber); }
.mc-red::before    { background: var(--red); }
.mc-name   { font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
.mc-score  { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 600; line-height: 1; margin-bottom: .25rem; }
.mc-interp { font-size: .8rem; color: var(--text2); }
.mc-desc   { font-size: .72rem; color: var(--muted2); margin-top: .4rem; font-style: italic; }

/* ── NLI cards ── */
.nli-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    box-shadow: var(--shadow);
}
.nli-dir     { font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); margin-bottom: .6rem; }
.nli-result  { font-size: 1.5rem; font-weight: 700; margin-bottom: .35rem; }
.nli-conf    { font-size: .86rem; color: var(--muted); }

/* ── Info / verdict box ── */
.info-box {
    border-radius: var(--radius);
    padding: .85rem 1.1rem;
    font-size: .88rem;
    margin: 1rem 0;
    border: 1px solid;
}
.info-green { background: var(--accent-lt); border-color: #b8d8d2; color: var(--accent2); }
.info-amber { background: var(--amber-lt);  border-color: #fcd34d; color: #92400e; }
.info-red   { background: var(--red-lt);    border-color: #f5b7b1; color: var(--red); }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Summary table ── */
.sum-table { width: 100%; border-collapse: collapse; font-size: .87rem; }
.sum-table th {
    text-align: left;
    font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); padding: .5rem .75rem;
    border-bottom: 2px solid var(--border);
}
.sum-table td { padding: .6rem .75rem; border-bottom: 1px solid var(--surface2); color: var(--text2); }
.sum-table tr:last-child td { border-bottom: none; font-weight: 700; color: var(--text); }
.mono { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading bi-encoder …")
def load_bi_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading similarity cross-encoder …")
def load_sim_cross():
    return CrossEncoder("cross-encoder/stsb-roberta-base")

@st.cache_resource(show_spinner="Loading NLI cross-encoder …")
def load_nli_cross():
    # cross-encoder/nli-deberta-v3-small: fast, accurate 3-class NLI
    # Label order: 0 = contradiction, 1 = entailment, 2 = neutral
    return CrossEncoder("cross-encoder/nli-deberta-v3-small")

bi_encoder  = load_bi_encoder()
sim_cross   = load_sim_cross()
nli_cross   = load_nli_cross()


# ─────────────────────────────────────────────────────────────────────────────
# Similarity functions
# ─────────────────────────────────────────────────────────────────────────────
def char_similarity(t1, t2):
    return SequenceMatcher(None, t1, t2).ratio()

def jaccard_similarity(t1, t2):
    s1 = set(word_tokenize(t1.lower()))
    s2 = set(word_tokenize(t2.lower()))
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0

def tfidf_similarity(t1, t2):
    mat = TfidfVectorizer().fit_transform([t1, t2])
    return float(sk_cosine(mat[0], mat[1])[0][0])

def embedding_similarity(t1, t2):
    emb = bi_encoder.encode([t1, t2])
    return float(sk_cosine([emb[0]], [emb[1]])[0][0])

def cross_encoder_similarity(t1, t2):
    """stsb-roberta-base outputs a 0–5 score; normalise to [0, 1]."""
    raw = sim_cross.predict([(t1, t2)])[0]
    return float(np.clip(raw / 5.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# NLI via CrossEncoder  (no transformers pipeline → no parsing errors)
# ─────────────────────────────────────────────────────────────────────────────
NLI_LABELS = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]   # matches model label order

def nli_analysis(t1, t2):
    """
    Returns ((label, confidence), (label, confidence)) for A→B and B→A.
    apply_softmax=True converts raw logits to probabilities.
    """
    scores = nli_cross.predict([(t1, t2), (t2, t1)], apply_softmax=True)

    def _parse(row):
        idx = int(np.argmax(row))
        return NLI_LABELS[idx], float(row[idx])

    return _parse(scores[0]), _parse(scores[1])


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level cross-encoder analysis
# ─────────────────────────────────────────────────────────────────────────────
def sentence_level(t1, t2):
    sents1 = sent_tokenize(t1)
    sents2 = sent_tokenize(t2)
    if not sents1 or not sents2:
        return [], 0.0

    pairs  = [(s1, s2) for s1 in sents1 for s2 in sents2]
    raw    = sim_cross.predict(pairs)
    scores = np.clip(np.array(raw, dtype=float) / 5.0, 0.0, 1.0)
    matrix = scores.reshape(len(sents1), len(sents2))

    matches = []
    for i, s1 in enumerate(sents1):
        j     = int(np.argmax(matrix[i]))
        score = float(matrix[i][j])
        matches.append({
            "Sentence A":        s1[:95] + ("…" if len(s1) > 95 else ""),
            "Best Match in B":   sents2[j][:95] + ("…" if len(sents2[j]) > 95 else ""),
            "Score":             score,
            "Verdict":           interpret(score),
        })
    return matches, float(np.mean([m["Score"] for m in matches]))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def interpret(s):
    if s >= 0.85: return "Very high similarity"
    if s >= 0.70: return "Strong similarity"
    if s >= 0.50: return "Moderate similarity"
    if s >= 0.30: return "Weak similarity"
    return "Very low similarity"

def score_color(s):
    if s >= 0.70: return "var(--red)"
    if s >= 0.50: return "var(--amber)"
    return "var(--accent)"

def score_class(s):
    if s >= 0.70: return "mc-red"
    if s >= 0.50: return "mc-amber"
    return "mc-green"

def nli_display(label):
    label = label.upper()
    if label == "ENTAILMENT":    return "✅ Entailment",   "var(--accent)",  "info-green"
    if label == "CONTRADICTION": return "❌ Contradiction", "var(--red)",    "info-red"
    return "⚪ Neutral", "var(--muted)", "info-green"

def word_count(t):
    return len(t.split())


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  
  <h1 class="hero-title">🔬 Text Similarity &amp; Plagiarism Detector</h1>
  <p class="hero-sub">This App is designed to help you analyze and compare textual content for similarity and potential plagiarism.</p>
  <p class="hero-sub">Developed by: <strong>Al Yazdani</strong> </p>
  </div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Text inputs
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_A = (
    "In recent years, machine learning has emerged as a fundamental tool in "
    "quantitative research, enabling the extraction of complex patterns from "
    "large-scale datasets. Supervised learning techniques, in particular, have "
    "demonstrated strong performance in predictive modeling tasks, where labeled "
    "data is available. However, the effectiveness of these models depends "
    "critically on the quality of the input features and the assumptions "
    "underlying the training process. As a result, careful data preprocessing, "
    "feature engineering, and model validation are essential components of any "
    "robust machine learning pipeline."
)
DEFAULT_B = (
    "Over the past decade, machine learning has become an essential methodology "
    "in quantitative analysis, allowing researchers to uncover intricate "
    "structures within vast datasets. Among various approaches, supervised "
    "learning methods have proven especially effective for prediction problems "
    "that rely on labeled observations. Nevertheless, model performance is "
    "highly sensitive to the choice of input variables and the assumptions made "
    "during training. Consequently, rigorous data preparation, thoughtful "
    "feature construction, and thorough validation procedures play a crucial "
    "role in building reliable machine learning systems."
)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown('<div class="txt-label"><span class="badge">A</span> Text A</div>', unsafe_allow_html=True)
    text1 = st.text_area("Text A", DEFAULT_A, height=210, label_visibility="collapsed")
    wc1   = word_count(text1)
    st.markdown(f'<div class="wc{"  over" if wc1 > 500 else ""}">{wc1} / 500 words</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="txt-label"><span class="badge">B</span> Text B</div>', unsafe_allow_html=True)
    text2 = st.text_area("Text B", DEFAULT_B, height=210, label_visibility="collapsed")
    wc2   = word_count(text2)
    st.markdown(f'<div class="wc{"  over" if wc2 > 500 else ""}">{wc2} / 500 words</div>', unsafe_allow_html=True)

if wc1 > 500 or wc2 > 500:
    st.error("⚠️  Each text must be 500 words or fewer.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Weight configuration
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)

with st.expander("⚖️  Configure Metric Weights", expanded=False):
    st.caption("Weights are auto-normalized to 1.0. Higher weight = more influence on the Weighted Overall Score.")
    wc1_, wc2_ = st.columns(2, gap="large")
    with wc1_:
        w_char    = st.slider("Character similarity",          0.0, 1.0, 0.05, 0.05)
        w_jaccard = st.slider("Jaccard word overlap",          0.0, 1.0, 0.05, 0.05)
        w_tfidf   = st.slider("TF-IDF cosine",                 0.0, 1.0, 0.15, 0.05)
    with wc2_:
        w_embed   = st.slider("Bi-Encoder (semantic)",         0.0, 1.0, 0.25, 0.05)
        w_cross   = st.slider("Cross-Encoder (deep pairwise)", 0.0, 1.0, 0.50, 0.05)

    raw_sum = w_char + w_jaccard + w_tfidf + w_embed + w_cross
    if raw_sum == 0:
        st.error("At least one weight must be greater than 0.")
        st.stop()
    st.caption(f"Raw sum: **{raw_sum:.2f}** → auto-normalized to 1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    run = st.button("🔍  Run Analysis", use_container_width=True)

if not run:
    st.stop()

if not text1.strip() or not text2.strip():
    st.error("Please enter text in both boxes.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Compute all scores
# ─────────────────────────────────────────────────────────────────────────────
prog = st.progress(0, "Starting …")

prog.progress(10, "Computing character & Jaccard similarity …")
char = char_similarity(text1, text2)
jac  = jaccard_similarity(text1, text2)

prog.progress(25, "Computing TF-IDF cosine similarity …")
tfidf = tfidf_similarity(text1, text2)

prog.progress(38, "Running bi-encoder embeddings …")
embed = embedding_similarity(text1, text2)

prog.progress(52, "Running similarity cross-encoder …")
cross = cross_encoder_similarity(text1, text2)

prog.progress(68, "Running NLI (DeBERTa cross-encoder) …")
(fwd_label, fwd_conf), (bwd_label, bwd_conf) = nli_analysis(text1, text2)

prog.progress(84, "Sentence-level cross-encoder analysis …")
sent_matches, sent_avg = sentence_level(text1, text2)

prog.progress(100, "Done!")
prog.empty()

# Normalize weights
nw = np.array([w_char, w_jaccard, w_tfidf, w_embed, w_cross], dtype=float)
nw /= nw.sum()
nw_char, nw_jac, nw_tfidf, nw_embed, nw_cross = nw

weighted = nw_char*char + nw_jac*jac + nw_tfidf*tfidf + nw_embed*embed + nw_cross*cross


# ─────────────────────────────────────────────────────────────────────────────
# ② Individual metric cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)
st.markdown('<div class="sec-eyebrow">Similarity Scores</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📊 Individual Metrics</div>', unsafe_allow_html=True)

METRICS = [
    ("🔤 Character",     char,  "Character-level sequence overlap"),
    ("🔠 Jaccard",       jac,   "Word-level set overlap"),
    ("📈 TF-IDF",        tfidf, "Term-importance cosine similarity"),
    ("🧬 Bi-Encoder",    embed, "Semantic vector similarity"),
    ("🎯 Cross-Encoder", cross, "Deep pairwise scoring"),
]

r1 = st.columns(5, gap="medium")
#r2 = st.columns(2, gap="medium")
all_cols = list(r1) #+ list(r2)

for (name, score, desc), col in zip(METRICS, all_cols):
    clr = score_color(score)
    cls = score_class(score)
    with col:
        st.markdown(f"""
        <div class="mc {cls}">
          <div class="mc-name">{name}</div>
          <div class="mc-score" style="color:{clr};">{score:.1%}</div>
          <div class="mc-interp">{interpret(score)}</div>
          <div class="mc-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ① Overall score banner
# ─────────────────────────────────────────────────────────────────────────────
oc = score_color(weighted)
bg_map = {"var(--red)": "var(--red-lt)", "var(--amber)": "var(--amber-lt)", "var(--accent)": "var(--accent-lt)"}
bg = bg_map.get(oc, "var(--accent-lt)")
border_map = {"var(--red)": "#f5b7b1", "var(--amber)": "#fcd34d", "var(--accent)": "#b8d8d2"}
bc = border_map.get(oc, "#b8d8d2")

st.markdown(f"""
<div class="overall-card" style="background:{bg}; border-color:{bc};">
  <p class="overall-pct" style="color:{oc};">{weighted:.1%}</p>
  <p class="overall-verdict" style="color:{oc};">
    ⭐ Weighted Overall Similarity — <em>{interpret(weighted)}</em>
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ③ NLI results
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)
st.markdown('<div class="sec-eyebrow">Logical Relationship</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🧠 Natural Language Inference (NLI)</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sec-body">Determines the <em>logical relationship</em> between the two texts. Unlike similarity scores, NLI can detect when '
    'paraphrased texts contradict each other in meaning.</p>',
    unsafe_allow_html=True
)

nli_c1, nli_c2 = st.columns(2, gap="large")
for col, lbl, conf, direction in [
    (nli_c1, fwd_label, fwd_conf, "A → B  (A as premise, B as hypothesis)"),
    (nli_c2, bwd_label, bwd_conf, "B → A  (B as premise, A as hypothesis)"),
]:
    emoji_txt, lbl_color, _ = nli_display(lbl)
    with col:
        st.markdown(f"""
        <div class="nli-card">
          <div class="nli-dir">{direction}</div>
          <div class="nli-result" style="color:{lbl_color};">{emoji_txt}</div>
          <div class="nli-conf">Confidence: <strong style="color:{lbl_color};">{conf:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

# NLI verdict
fl, bl = fwd_label.upper(), bwd_label.upper()
if fl == "ENTAILMENT" and bl == "ENTAILMENT":
    verdict_cls  = "info-amber"
    verdict_html = "🟠 <strong>Mutual entailment</strong> — texts are semantically equivalent or near-paraphrases. Strong plagiarism signal."
elif "CONTRADICTION" in (fl, bl):
    verdict_cls  = "info-green"
    verdict_html = "🟢 <strong>Contradiction detected</strong> — despite surface similarity, the texts express opposing ideas in at least one direction."
elif "ENTAILMENT" in (fl, bl):
    verdict_cls  = "info-amber"
    verdict_html = "🟡 <strong>One-way entailment</strong> — one text's meaning is contained within the other."
else:
    verdict_cls  = "info-green"
    verdict_html = "🟢 <strong>Neutral</strong> — texts are logically independent; shared vocabulary does not imply shared meaning."

st.markdown(f'<div class="info-box {verdict_cls}">{verdict_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ④ Sentence-level analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)
st.markdown('<div class="sec-eyebrow">Granular Breakdown</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">🔬 Sentence-Level Cross-Encoder Analysis</div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sec-body">Each sentence in <strong>Text A</strong> is matched to its closest counterpart '
    f'in <strong>Text B</strong>. Average match score: <strong>{sent_avg:.1%}</strong> '
    f'across {len(sent_matches)} sentence(s).</p>',
    unsafe_allow_html=True
)

if sent_matches:
    df_sent = pd.DataFrame(sent_matches)

    def _color_score(val):
        if isinstance(val, float):
            c = score_color(val)
            bg_map2 = {"var(--red)": "#fdf0ee", "var(--amber)": "#fffbeb", "var(--accent)": "#e8f4f1"}
            return f"background-color:{bg_map2.get(c,'#e8f4f1')}; font-weight:600; font-family:'JetBrains Mono',monospace;"
        return ""

    styled = (
        df_sent.style
        .applymap(_color_score, subset=["Score"])
        .format({"Score": "{:.1%}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Summary panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)
st.markdown('<div class="sec-eyebrow">Report</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">📋 Analysis Summary</div>', unsafe_allow_html=True)

sum_c1, sum_c2 = st.columns([3, 2], gap="large")

with sum_c1:
    rows = [
        ("Character",       char,     nw_char,   True),
        ("Jaccard",         jac,      nw_jac,    True),
        ("TF-IDF",          tfidf,    nw_tfidf,  True),
        ("Bi-Encoder",      embed,    nw_embed,  True),
        ("Cross-Encoder",   cross,    nw_cross,  True),
        ("Weighted Overall",weighted, 1.0,       False),
    ]
    trs = ""
    for name, sc, wt, show_wt in rows:
        c     = score_color(sc)
        wt_s  = f"{wt:.0%}" if show_wt else "—"
        trs  += (
            f"<tr><td>{name}</td>"
            f"<td class='mono' style='color:{c};'>{sc:.1%}</td>"
            f"<td>{interpret(sc)}</td>"
            f"<td class='mono' style='color:var(--muted);'>{wt_s}</td></tr>"
        )
    st.markdown(f"""
    <table class="sum-table">
      <thead><tr><th>Metric</th><th>Score</th><th>Interpretation</th><th>Weight</th></tr></thead>
      <tbody>{trs}</tbody>
    </table>
    """, unsafe_allow_html=True)

with sum_c2:
    fwd_e, fwd_c, _ = nli_display(fwd_label)
    bwd_e, bwd_c, _ = nli_display(bwd_label)
    st.markdown(f"""
    <div style="background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.3rem; box-shadow:var(--shadow);">
      <div class="sec-eyebrow" style="margin-bottom:.7rem;">NLI Findings</div>
      <div style="margin-bottom:.6rem;">
        <span style="font-size:.75rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.07em;">A → B</span><br>
        <span style="color:{fwd_c}; font-weight:700;">{fwd_e}</span>
        <span class="mono" style="color:var(--muted); font-size:.82rem;"> {fwd_conf:.1%}</span>
      </div>
      <div style="margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid var(--border);">
        <span style="font-size:.75rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.07em;">B → A</span><br>
        <span style="color:{bwd_c}; font-weight:700;">{bwd_e}</span>
        <span class="mono" style="color:var(--muted); font-size:.82rem;"> {bwd_conf:.1%}</span>
      </div>
      <div class="sec-eyebrow" style="margin-bottom:.6rem;">Sentence Analysis</div>
      <div style="font-size:.88rem; color:var(--text2); margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid var(--border);">
        Avg match: <strong style="color:var(--accent);">{sent_avg:.1%}</strong><br>
        Sentences: <strong>{len(sent_matches)}</strong>
      </div>
      <div class="sec-eyebrow" style="margin-bottom:.4rem;">Overall Verdict</div>
      <strong style="font-size:1rem; color:{score_color(weighted)};">{interpret(weighted)}</strong>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ CSV export
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)

buf = StringIO()
buf.write("=== INPUT TEXTS ===\n")
buf.write(f'"Text A","{text1.replace(chr(34), chr(39))}"\n')
buf.write(f'"Text B","{text2.replace(chr(34), chr(39))}"\n\n')

buf.write("=== SIMILARITY SCORES ===\n")
pd.DataFrame({
    "Metric":             ["Character", "Jaccard", "TF-IDF", "Bi-Encoder", "Cross-Encoder", "Weighted Overall"],
    "Score":              [char, jac, tfidf, embed, cross, weighted],
    "Interpretation":     [interpret(s) for s in [char, jac, tfidf, embed, cross, weighted]],
    "Normalized Weight":  [f"{nw_char:.2%}", f"{nw_jac:.2%}", f"{nw_tfidf:.2%}", f"{nw_embed:.2%}", f"{nw_cross:.2%}", "—"],
}).to_csv(buf, index=False)

buf.write("\n=== NLI RESULTS ===\n")
pd.DataFrame({
    "Direction": ["A→B", "B→A"],
    "Label":     [fwd_label, bwd_label],
    "Confidence":[fwd_conf,  bwd_conf],
}).to_csv(buf, index=False)

if sent_matches:
    buf.write("\n=== SENTENCE-LEVEL ANALYSIS ===\n")
    pd.DataFrame(sent_matches).to_csv(buf, index=False)

dl_col, _ = st.columns([1, 3])
with dl_col:
    st.download_button(
        "📥  Download Full Report (CSV)",
        data=buf.getvalue(),
        file_name="similarity_report_v3.csv",
        mime="text/csv",
        use_container_width=True,
    )
