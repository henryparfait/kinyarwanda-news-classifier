"""Kinyarwanda News Topic Classifier: Streamlit web interface.

Model loading and prediction live in predictor.py.
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

st.set_page_config(page_title="Kinyarwanda News Topic Classifier", page_icon="📰", layout="centered")


@st.cache_resource(show_spinner="Loading the model… (the first load can take about a minute)")
def get_model():
    return load_model()


tokenizer, model, id2label = get_model()
categories = ", ".join(id2label[i] for i in sorted(id2label))

st.title("📰 Kinyarwanda News Topic Classifier")
st.write("Paste a **Kinyarwanda news article** (title + text) and the model predicts its topic "
         f"among 14 categories: {categories}.")

if "article" not in st.session_state:
    st.session_state.article = ""


def use_example():
    choice = st.session_state.example
    if choice in EXAMPLE_NAMES:
        st.session_state.article = EXAMPLES[EXAMPLE_NAMES.index(choice)]["text"]


st.selectbox("Try an example", [NO_EXAMPLE] + EXAMPLE_NAMES, key="example", on_change=use_example)
text = st.text_area("Kinyarwanda news article", key="article", height=260,
                    placeholder="Andika cyangwa ushyiremo inkuru hano… (paste an article here)")

if st.button("Classify", type="primary"):
    if len(text.split()) < 5:
        st.warning("Please paste a longer Kinyarwanda news text (at least one or two sentences).")
    else:
        with st.spinner("Classifying…"):
            ranked, n_tokens = classify(text, tokenizer, model, id2label)
        label, conf = ranked[0]
        st.subheader(f"Predicted category: {label}")
        st.caption(f"Confidence: {conf:.1%}")
        if conf < LOW_CONFIDENCE:
            st.warning("Low confidence. The text may cover several topics, belong to none of the 14 "
                       "categories, or not be in Kinyarwanda. Treat this prediction with caution.")
        if n_tokens > MAX_LEN - 2:
            st.info(f"The text is {n_tokens} tokens long; the model only reads the first {MAX_LEN - 2}.")
        st.markdown("**Top-5 categories**")
        for name, p in ranked[:5]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{name}: {p:.1%}")

with st.expander("About this model"):
    st.markdown(f"""
- **Model:** `{INFO['base_model']}` fine-tuned on the KINNEWS corpus (Niyongabo et al., 2020): experiment {INFO['exp_id']}, max {MAX_LEN} tokens, class-weighted loss. Hosted on the Hugging Face Hub: `{INFO['model_repo']}`.
- **How the app works:** the model is downloaded once and cached; your text is tokenised, passed through the model, and the softmax probabilities are shown.
- **Test results (deduplicated KINNEWS test set, {INFO['test_size']} articles):** accuracy {INFO['test_accuracy']:.1%}, macro-F1 {INFO['test_macro_f1']:.3f}.
- **Limitations:** trained on Rwandan online news from about 2019–2020; rare categories (tourism, fashion, environment, education) are much less reliable; the model can partly rely on source names and bylines; non-news or non-Kinyarwanda text gives unreliable outputs.
- **Code and report:** {INFO['github_url']}
""")
