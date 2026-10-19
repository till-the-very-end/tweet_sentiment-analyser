# Sumo — Extractive Document Summarizer (Information Retrieval project)

Flask + NumPy backend, minimal vanilla-JS frontend.

## Run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # open http://127.0.0.1:5000
pytest                           # run tests
```

## Structure
```
app.py                  Flask routes: /, /api/summarize, /api/rouge
summarizer/core.py      IR engine (all algorithms live here)
static/index.html       UI (calls the API)
tests/                  pytest unit tests
```

## Algorithms (summarizer/core.py)
| Step | Technique |
|---|---|
| Preprocess | sentence split, tokenize, stopwords, stemming |
| Representation | TF-IDF term x sentence matrix, cosine similarity |
| TextRank | PageRank (d=0.85) on the similarity graph |
| LSA | SVD (numpy.linalg.svd), Steinberger-Jezek scoring |
| Query-focused | Okapi BM25 (k1=1.5, b=0.75) |
| Redundancy | Maximal Marginal Relevance (lambda=0.7) |
| Evaluation | ROUGE-1 / ROUGE-2 (`POST /api/rouge`) |

## Ideas to extend
- Replace `stem()` with NLTK Porter/Snowball; add PDF input (pypdf)
- Sentence embeddings (sentence-transformers) as a dense-retrieval scorer
- Abstractive stage with a seq2seq model (BART/T5); evaluate with ROUGE on CNN/DailyMail
- Multi-document summarization and an inverted index for search over a corpus
