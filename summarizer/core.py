"""Extractive summarization engine built from classic Information Retrieval techniques.

Pipeline: preprocess -> TF-IDF vector space -> score (TextRank / TF-IDF / LSA / BM25)
          -> select (top-k or MMR) -> reorder.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its "
    "this that these those he she they we you i not no so than then there their his her our your my me "
    "him them which who whom what when where while about into over under after before also can could "
    "would should will may might do does did have has had just more most such only other any each very "
    "however".split()
)
_SUFFIX = re.compile(r"(ingly|edly|ing|ed|es|s|ly)$")


def stem(word: str) -> str:
    """Light suffix-stripping stemmer (swap for nltk's PorterStemmer to improve accuracy)."""
    return _SUFFIX.sub("", word) if len(word) > 4 else word


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\d])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def tokenize(sentence: str, surface: dict | None = None) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]{2,}", sentence.lower()):
        if w in STOPWORDS:
            continue
        s = stem(w)
        if surface is not None:
            surface[s][w] += 1
        out.append(s)
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    m = x.max() if x.size else 0
    return x / m if m > 0 else x


def build_tfidf(tokens: list[list[str]]):
    """Returns (vocab, idf, A) where A is the terms x sentences TF-IDF matrix."""
    n = len(tokens)
    df = Counter(w for t in tokens for w in set(t))
    vocab = {w: i for i, w in enumerate(sorted(df))}
    idf = {w: math.log((n + 1) / (df[w] + 1)) + 1 for w in vocab}
    A = np.zeros((len(vocab), n))
    for j, t in enumerate(tokens):
        for w, c in Counter(t).items():
            A[vocab[w], j] = c / len(t) * idf[w]
    return vocab, idf, A


def similarity_matrix(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1
    An = A / norms
    S = An.T @ An
    np.fill_diagonal(S, 0)
    return S


def tfidf_scores(A, tokens):
    lengths = np.array([max(len(t), 1) for t in tokens])
    s = A.sum(axis=0) / np.sqrt(lengths)
    s[0] *= 1.25  # lead-sentence position bias
    return _norm(s)


def textrank_scores(S, d=0.85, iters=100, tol=1e-8):
    n = S.shape[0]
    rows = S.sum(axis=1)
    rows[rows == 0] = 1
    M = S / rows[:, None]
    r = np.full(n, 1 / n)
    for _ in range(iters):
        new = (1 - d) / n + d * M.T @ r
        done = np.abs(new - r).sum() < tol
        r = new
        if done:
            break
    return _norm(r)


def lsa_scores(A, k=3):
    """Steinberger-Jezek: sentence length in the top-k singular subspace."""
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(s))
    return _norm(np.sqrt(((s[:k, None] * Vt[:k]) ** 2).sum(axis=0)))


def bm25_scores(query_terms, tokens, idf, k1=1.5, b=0.75):
    avg = np.mean([len(t) for t in tokens]) or 1
    out = np.zeros(len(tokens))
    for j, t in enumerate(tokens):
        c = Counter(t)
        for w in query_terms:
            f = c.get(w, 0)
            out[j] += idf.get(w, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avg))
    return _norm(out)


def mmr_select(scores, S, k, lam=0.7):
    """Maximal Marginal Relevance: balance relevance against redundancy."""
    chosen = []
    while len(chosen) < k:
        best, best_val = None, -1e9
        for i in range(len(scores)):
            if i in chosen:
                continue
            red = max((S[i][j] for j in chosen), default=0)
            val = lam * scores[i] - (1 - lam) * red
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
    return sorted(chosen)


def summarize(text: str, method: str = "hybrid", percent: int = 25, query: str = ""):
    sentences = split_sentences(text)
    n = len(sentences)
    if n < 2:
        return None
    surface = defaultdict(Counter)
    tokens = [tokenize(s, surface) for s in sentences]
    vocab, idf, A = build_tfidf(tokens)
    S = similarity_matrix(A)

    tf, tr, ls = tfidf_scores(A, tokens), textrank_scores(S), lsa_scores(A)
    scores = {"tfidf": tf, "textrank": tr, "lsa": ls}.get(method, (tf + tr + ls) / 3)

    q = tokenize(query)
    if q:
        scores = 0.4 * scores + 0.6 * bm25_scores(q, tokens, idf)

    k = max(1, round(n * percent / 100))
    if method == "hybrid":
        pick = mmr_select(scores, S, k)
    else:
        pick = sorted(np.argsort(-scores)[:k].tolist())

    weights = A.sum(axis=1)
    inv = {i: w for w, i in vocab.items()}
    kw = [surface[inv[i]].most_common(1)[0][0] for i in np.argsort(-weights)[:10]]
    return {"S": sentences, "sc": scores.round(4).tolist(), "pick": pick, "kw": kw, "n": n}


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    """ROUGE-N precision / recall / F1 on lowercase word n-grams."""
    def grams(t):
        w = re.findall(r"[a-z0-9]+", t.lower())
        return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    c, r = grams(candidate), grams(reference)
    overlap = sum((c & r).values())
    p = overlap / max(sum(c.values()), 1)
    rec = overlap / max(sum(r.values()), 1)
    f = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"precision": p, "recall": rec, "f1": f}
