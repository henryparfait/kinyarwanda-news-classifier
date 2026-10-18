# Kinyarwanda News Topic Classification

Automatically classifying Kinyarwanda news articles into 14 topics (politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, relationships), comparing classical machine learning, recurrent neural networks and African-centric pretrained Transformers.

**Individual NLP capstone project · Option 3: Text Classification for an African Language**

## Project links

| Resource | Link |
|---|---|
| 🌐 Live demo (Streamlit) | https://kinyarwandanews.streamlit.app/ |
| 🤗 Fine-tuned model | https://huggingface.co/henryparfait/kinnews-topic-classifier |
| 🎥 Demo video | [TODO: video link] |
| 📄 Report (PDF) | [`report/report.pdf`](report/report.pdf) |

> The free hosting puts the app to sleep after a period without visitors. Opening the link wakes it up in about a minute. If it is unavailable, the app can be run locally (see [Running the app locally](#running-the-app-locally)).

## The problem

Kinyarwanda is spoken by more than 12 million people, mostly in Rwanda, and Rwandan news sites publish a large volume of content in it every day. Automatic topic classification is a basic building block for organising that content: news aggregation, search, recommendation and monitoring. Yet most text classification tools are built for high-resource languages. Kinyarwanda is morphologically rich (a single word can carry subject, tense, object and verb root, e.g. *ntibazabibona*), so a model that treats every word form as a separate token sees very sparse data.

My goal was to find out which approach works best for Kinyarwanda news topic classification with a limited amount of labelled data, and to understand *why*.

## Dataset

I used **KINNEWS** (Niyongabo et al., 2020), a corpus of Kinyarwanda news articles collected from Rwandan news websites and labelled with 14 categories (MIT license).

While preparing the data I found several quality problems:

| Finding | Numbers |
|---|---|
| Exact duplicate articles in the raw training set | 17,014 rows → **7,630 unique articles** (9,387 repeated URLs; one article appeared 80 times) |
| Duplicates with **conflicting labels** | 156 articles (removed) |
| Duplicates in the test set | 4,254 rows → 3,464 unique articles |
| Severe class imbalance | *politic* 1,591 training articles vs. *tourism* 30 |
| **Train/test label distribution shift** | e.g. *environment*: 34 train vs. 100 test; *history*: 196 train vs. 11 test |

**Final splits:** 6,726 training / 745 validation (stratified 10% of training) / 3,464 test (deduplicated official test set). Results on the original 4,254-row test set are also logged, so they can be compared with the original paper.

Preprocessing was kept light: HTML entity decoding, Unicode normalisation, **apostrophe standardisation** (important for Kinyarwanda contractions such as *n'*, *y'*, *by'*), URL/e-mail removal, and joining title and body.

## Methodology and experiments

Each experiment changes **one thing** from the previous one, so every difference in score has a clear cause. Hyperparameters and the final model were selected on the **validation** set. The test set was used once per experiment. The main metric is **macro-F1**, because with this level of imbalance accuracy is dominated by the largest classes.

| Exp | Model | What changed | Val macro-F1 | Test acc. | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|---|
| B0 | Majority class | n/a (floor) | 0.027 | 0.091 | 0.012 | 0.015 |
| B1 | Word TF-IDF (1–2 grams) + Logistic Regression | n/a | 0.553 | 0.691 | 0.413 | 0.656 |
| B2 | Char TF-IDF (2–5 grams) + Logistic Regression | words → character n-grams | 0.623 | 0.710 | 0.458 | 0.681 |
| B3 | B2 + balanced class weights | class weighting | 0.745 | 0.743 | 0.577 | 0.742 |
| N1 | BiLSTM, random embeddings, max pooling | n/a | 0.530 | 0.576 | 0.354 | 0.572 |
| N2 | BiLSTM, Word2Vec embeddings | embedding initialisation | 0.544 | 0.595 | 0.383 | 0.594 |
| N3 | BiLSTM, Word2Vec, attention pooling | pooling | 0.631 | 0.685 | 0.513 | 0.687 |
| T1 | AfriBERTa-base fine-tuned, 256 tokens | n/a | 0.686 | 0.748 | 0.578 | 0.742 |
| T2 | AfroXLMR-base fine-tuned, 256 tokens | pretrained model | 0.665 | **0.773** | **0.618** | **0.771** |
| **T3** | **AfriBERTa-base fine-tuned, 512 tokens (final model)** | context length | **0.691** | 0.755 | 0.581 | 0.751 |

Full log: [`results/experiment_log.csv`](results/experiment_log.csv)

### What I learned

- **Class weighting was the single most effective change** (B2 → B3: +0.12 macro-F1), because rare classes are heavily under-represented in training but common in the test set.
- **Character n-grams beat word features** (B1 → B2). The training set has over 140,000 distinct word forms in only 6,726 articles, and 7.8% of test tokens never appear in the training vocabulary. Character n-grams share information across forms of the same root.
- **Word2Vec embeddings trained on the corpus helped the BiLSTM learn faster** (validation macro-F1 after one epoch: 0.33 vs. 0.16), and **attention pooling added +0.13 macro-F1**. The BiLSTMs still trailed the weighted baseline, partly because they were trained without class weights; their fair comparison is B2, which N3 beats.
- **AfriBERTa's tokenizer fits Kinyarwanda better** than AfroXLMR's (1.88 vs. 2.49 sub-word tokens per word), despite a 3.5× smaller vocabulary.
- **Reading 512 instead of 256 tokens barely helped** (+0.003 macro-F1 for twice the training time). The topic of a news article is usually clear from its title and opening.
- **Pretrained Transformers only modestly beat a well-tuned TF-IDF model.** Topic classification is largely keyword-driven, and the train/test distribution shift limits every model regardless of architecture.
- **Validation and test disagreed:** T3 was best on validation, but T2 was best on test. The validation set contains only 3–4 articles for some classes, so its macro-F1 is noisy. I kept T3 as the final model because it was selected by the validation protocol (choosing by test score would leak test information), and I report T2's result openly.

## Error analysis (summary)

Details are in the report and in [`notebooks/05_error_analysis.ipynb`](notebooks/05_error_analysis.ipynb).

- **Rare classes fail most:** education (F1 0.07), history (0.16), environment (0.32) and culture (0.40), compared with sport (0.95) and entertainment (0.85).
- **Overlapping categories:** education → relationship (46% of education), environment → economy (30%) and → health (27%), fashion → entertainment (30%), culture → history (28%), economy → technology (13%).
- **Shortcut hypothesis tested and not supported:** the baseline's top features include source names and bylines (*muhabura*, *yanditswe na* = "written by"), but deleting them from every test article changed macro-F1 by only +0.003 (baseline) and 0.000 (Transformer). The models rely on content.
- **The official split is source-dependent:** *muhabura* appears in ~40% of *training* articles in economy, health and history but in almost none of the *test* articles in those classes, while it dominates test sport, entertainment and relationship articles. This explains the train/test distribution shift.
- **Likely label noise:** the Transformer's most confident errors are football articles (Amavubi, APR FC, Women's World Cup) labelled *entertainment* and predicted *sport* with 99% confidence; the model is right and the label is wrong.
- **Errors are shared across model families:** all three models fail on the same 15.7% of test articles, which points to the data rather than model capacity. Long articles (>600 words) are harder for every model, including the baseline that reads the full text, so truncation is not the main cause.
- **Confidence is informative:** predictions above 90% confidence are 86.8% accurate, below 50% only 29.5%, which motivates the app's low-confidence warning.
- **Spelling variation:** Word2Vec places variants next to each other (*abanyeshuri / abanyeshuli*, *amafaranga / amafranga*, *perezida / prezida*), which shows how much non-standard spelling the data contains.
- [TODO: manual review of N errors: X label ambiguity, Y likely mislabelled, …]

## Deployment

The final model (T3) is hosted on the **Hugging Face Hub**, and the web interface is a **Streamlit** app on **Streamlit Community Cloud**, deployed directly from this repository. (I first targeted Hugging Face Spaces, but Gradio Spaces on free hardware now require a paid plan.)

The app is split into two files: `predictor.py` loads the model and tokenizer from the Hub (cached after the first load) and runs prediction, and `streamlit_app.py` is the interface. For each request the text is tokenised (truncated to 512 tokens), passed through the model, and the softmax probabilities are shown as the top-5 categories. The app warns when confidence is below 50% and when an article was truncated. It also offers examples of correct predictions, known errors and an out-of-domain (English) input.

## Repository structure

```
├── notebooks/
│   ├── 01_data_preparation.ipynb   # download, cleaning, deduplication, splits, statistics
│   ├── 02_baselines.ipynb          # B0–B3: majority class, TF-IDF + Logistic Regression
│   ├── 03_bilstm.ipynb             # N1–N3: Word2Vec, BiLSTM, attention
│   ├── 04_transformers.ipynb       # T1–T3: AfriBERTa / AfroXLMR fine-tuning
│   ├── 05_error_analysis.ipynb     # per-class, confusion, confidence, shortcut test
│   └── 06_deploy.ipynb             # Hub upload + Streamlit app files
├── app/                            # Streamlit app (streamlit_app.py, predictor.py, requirements, examples, model info)
├── results/                        # experiment log, classification reports, analysis tables
├── figures/                        # plots used in the report
├── report/report.pdf
├── requirements.txt
└── LICENSE
```

## How to reproduce

All notebooks are designed for **Google Colab** and save their outputs to Google Drive (`MyDrive/kinyarwanda-news-classifier/`), so every stage can be re-run independently.

1. Open each notebook in Colab (*File → Upload notebook*) and run them **in order**, 01 → 06.
2. Runtimes: 01, 02 and 06 on **CPU**; 03 and 04 on a **T4 GPU**; 05 on either (GPU is faster).
3. Notebook 01 downloads KINNEWS automatically, so no manual download is needed.
4. Notebook 06 needs a Hugging Face access token with *write* permission (entered at runtime, never stored in code). The web app is then deployed on [Streamlit Community Cloud](https://share.streamlit.io) with main file `app/streamlit_app.py`.

Approximate run times on a T4: data prep 2 min · baselines 20 min · BiLSTMs 10 min (plus 6 min Word2Vec) · Transformers 35 min. All experiments use random seed 42.

### Running the app locally

```bash
git clone https://github.com/henryparfait/kinyarwanda-news-classifier.git
cd kinyarwanda-news-classifier/app
pip install -r requirements.txt
streamlit run streamlit_app.py      # then open http://localhost:8501
```

## What I implemented vs. what I used

**Implemented myself:** the data cleaning, deduplication and label-conflict audit; train/validation splitting; all training and evaluation code (including the BiLSTM model with masked max/attention pooling, the Transformer fine-tuning loop with class-weighted loss, warm-up scheduling and mixed precision); the experiment design; the tokenizer, shortcut, confidence and error analyses; the manual error review; and the Streamlit web app.

**Existing resources used:** the KINNEWS dataset; the pretrained AfriBERTa and AfroXLMR models; the scikit-learn, gensim, PyTorch, Hugging Face Transformers and Streamlit libraries; Hugging Face Hub and Streamlit Community Cloud for hosting.

## Limitations

- The KINNEWS train and test splits have very different class distributions, so the scores depend heavily on how well rare classes transfer.
- The validation set has only 3–4 examples for some classes, which makes model selection noisy.
- The articles come from a limited set of news sites (around 2019–2020), and models partly learn source-specific words.
- Each configuration was trained with one random seed, so small differences between models may not be significant.

## References

- Niyongabo, R. A., Hong, Q., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *COLING 2020*. https://doi.org/10.18653/v1/2020.coling-main.480
- Adelani, D. I., et al. (2023). MasakhaNEWS: News topic classification for African languages. *IJCNLP-AACL 2023*.
- Nzeyimana, A., & Niyongabo Rubungo, A. (2022). KinyaBERT: A morphology-aware Kinyarwanda language model. *ACL 2022*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*(11), 665–673.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *Proceedings of the 1st Workshop on Multilingual Representation Learning*. (AfriBERTa)
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*. (AfroXLMR)
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*. (XLM-R)
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8).
- Yang, Z., et al. (2016). Hierarchical attention networks for document classification. *NAACL 2016*.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020: System Demonstrations*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*.
- Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. *LREC 2010 Workshop*. (gensim)
- Streamlit (2024). Streamlit documentation. https://docs.streamlit.io

## Acknowledgements

AI assistance (Claude) was used to help structure code and documentation. I ran, tested and checked all code and results myself, and I am responsible for the work submitted.

## License

Code: MIT. The KINNEWS dataset is distributed under the MIT license by its authors; the pretrained models are subject to their own licenses on the Hugging Face Hub.
