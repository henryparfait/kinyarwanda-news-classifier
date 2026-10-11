"""Model loading and prediction for the Kinyarwanda News Topic Classifier.

This module is the bridge between the web interface and the ML model:
the fine-tuned Transformer is downloaded from the Hugging Face Hub, each input
text is tokenised (truncated to MAX_LEN tokens), passed through the model, and
softmax probabilities over the 14 KINNEWS categories are returned, ranked.
"""
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "model_info.json"), encoding="utf-8") as f:
    INFO = json.load(f)
MODEL_ID = INFO["model_repo"]
MAX_LEN = int(INFO["max_len"])


def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label


def classify(text, tokenizer, model, id2label):
    """Return ([(category, probability), ...] sorted high to low, number of tokens in the text)."""
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    ranked = sorted(((id2label[i], float(p)) for i, p in enumerate(probs)), key=lambda x: x[1], reverse=True)
    return ranked, n_tokens
