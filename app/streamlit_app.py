"""Kinyarwanda News Topic Classifier: Streamlit web interface.

Model loading and prediction live in predictor.py; this file only handles the interface.
"""
import json
import os

import streamlit as st

from predictor import INFO, MAX_LEN, classify, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "examples.json"), encoding="utf-8") as f:
    EXAMPLES = json.load(f)
LOW_CONFIDENCE = 0.5
NO_EXAMPLE = "Type or paste your own article"
EXAMPLE_NAMES = [f"{i + 1}. {e['description']}" for i, e in enumerate(EXAMPLES)]

# One icon and one colour per category, used in the header, result card and bars.
CATEGORY_STYLE = {
    "politic": ("🏛️", "#2563eb"),
    "sport": ("⚽", "#16a34a"),
    "economy": ("💰", "#ca8a04"),
    "health": ("🩺", "#dc2626"),
    "entertainment": ("🎬", "#db2777"),
    "history": ("📜", "#a16207"),
    "technology": ("💻", "#0891b2"),
    "tourism": ("🏞️", "#059669"),
    "culture": ("🥁", "#ea580c"),
    "fashion": ("👗", "#c026d3"),
    "religion": ("⛪", "#7c3aed"),
    "environment": ("🌿", "#65a30d"),
    "education": ("🎓", "#4f46e5"),
    "relationship": ("💞", "#e11d48"),
}


def style_of(label):
    return CATEGORY_STYLE.get(label.lower(), ("📰", "#6b7280"))


st.set_page_config(page_title="Kinyarwanda News Topic Classifier", page_icon="📰", layout="centered")

# Styling only. Colours use transparency so the page works in both light and dark mode.
st.markdown("""<style>
.hero {background: linear-gradient(120deg, #0077a8 0%, #20603d 100%); color: #fff; padding: 1.6rem 1.8rem 1.3rem;
       border-radius: 16px; border-bottom: 5px solid #fad201; margin-bottom: 1.3rem;}
.hero h1 {color: #fff; font-size: 1.85rem; margin: 0 0 .35rem 0; padding: 0;}
.hero p {color: #fff; margin: 0; opacity: .92; font-size: 1rem;}
.chips {margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .35rem;}
.chip {background: rgba(255,255,255,.18); border-radius: 999px; padding: .12rem .6rem; font-size: .8rem; white-space: nowrap;}
.result {border-radius: 14px; padding: 1rem 1.3rem; margin: .9rem 0 1rem; border: 1px solid rgba(128,128,128,.25);
         border-left: 7px solid var(--accent); background: rgba(128,128,128,.07);}
.result .label {font-size: 1.7rem; font-weight: 700; margin: .1rem 0;}
.result .sub {opacity: .75; font-size: .9rem;}
.bar-row {margin: .5rem 0;}
.bar-head {display: flex; justify-content: space-between; font-size: .93rem; margin-bottom: .2rem;}
.bar-track {background: rgba(128,128,128,.18); border-radius: 999px; height: 10px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.footer {text-align: center; opacity: .65; font-size: .85rem; margin-top: 2.2rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model… (the first load can take about a minute)")
def get_model():
    return load_model()


tokenizer, model, id2label = get_model()
labels = [id2label[i] for i in sorted(id2label)]

chips = "".join(f'<span class="chip">{style_of(c)[0]} {c}</span>' for c in labels)
st.markdown(
    '<div class="hero"><h1>📰 Kinyarwanda News Topic Classifier</h1>'
    '<p>Paste a Kinyarwanda news article (title + text) and the model predicts its topic.</p>'
    f'<div class="chips">{chips}</div></div>',
    unsafe_allow_html=True,
)

if "article" not in st.session_state:
    st.session_state.article = ""


def use_example():
    choice = st.session_state.example
    if choice in EXAMPLE_NAMES:
        st.session_state.article = EXAMPLES[EXAMPLE_NAMES.index(choice)]["text"]


def clear_text():
    st.session_state.article = ""
    st.session_state.example = NO_EXAMPLE


st.selectbox("Try an example", [NO_EXAMPLE] + EXAMPLE_NAMES, key="example", on_change=use_example)
text = st.text_area("Kinyarwanda news article", key="article", height=240,
                    placeholder="Andika cyangwa ushyiremo inkuru hano… (paste an article here)")
st.caption(f"{len(text.split())} words")

col_classify, col_clear, _ = st.columns([1.3, 1, 3])
run = col_classify.button("Classify", type="primary")
col_clear.button("Clear", on_click=clear_text)

if run:
    if len(text.split()) < 5:
        st.warning("Please paste a longer Kinyarwanda news text (at least one or two sentences).")
    else:
        with st.spinner("Classifying…"):
            ranked, n_tokens = classify(text, tokenizer, model, id2label)
        label, conf = ranked[0]
        icon, color = style_of(label)
        st.markdown(
            f'<div class="result" style="--accent:{color}"><div class="sub">Predicted category</div>'
            f'<div class="label">{icon} {label.capitalize()}</div>'
            f'<div class="sub">Confidence: <b>{conf:.1%}</b></div></div>',
            unsafe_allow_html=True,
        )
        if conf < LOW_CONFIDENCE:
            st.warning("Low confidence. The text may cover several topics, belong to none of the 14 "
                       "categories, or not be in Kinyarwanda. Treat this prediction with caution.")
        if n_tokens > MAX_LEN - 2:
            st.info(f"The text is {n_tokens} tokens long; the model only reads the first {MAX_LEN - 2}.")

        st.markdown("**Top-5 categories**")
        bars = ""
        for name, p in ranked[:5]:
            bar_icon, bar_color = style_of(name)
            bars += (f'<div class="bar-row"><div class="bar-head"><span>{bar_icon} {name}</span><span>{p:.1%}</span></div>'
                     f'<div class="bar-track"><div class="bar-fill" style="width:{max(p * 100, 0.5):.1f}%;'
                     f'background:{bar_color}"></div></div></div>')
        st.markdown(bars, unsafe_allow_html=True)

with st.expander("About this model"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{INFO['test_accuracy']:.1%}")
    m2.metric("Macro-F1", f"{INFO['test_macro_f1']:.3f}")
    m3.metric("Max tokens", MAX_LEN)
    st.markdown(f"""
- **Model:** `{INFO['base_model']}` fine-tuned on the KINNEWS corpus (Niyongabo et al., 2020): experiment {INFO['exp_id']}, class-weighted loss. Hosted on the Hugging Face Hub: `{INFO['model_repo']}`.
- **How the app works:** the model is downloaded once and cached; your text is tokenised, passed through the model, and the softmax probabilities are shown.
- **Test set:** the deduplicated KINNEWS test set ({INFO['test_size']} articles).
- **Limitations:** trained on Rwandan online news from about 2019–2020; rare categories (tourism, fashion, environment, education) are much less reliable; non-news or non-Kinyarwanda text gives unreliable outputs.
""")

st.markdown(
    f'<div class="footer">Built for an NLP capstone project · '
    f'<a href="{INFO["github_url"]}" target="_blank">Code &amp; report</a> · '
    f'<a href="https://huggingface.co/{INFO["model_repo"]}" target="_blank">Model</a></div>',
    unsafe_allow_html=True,
)
