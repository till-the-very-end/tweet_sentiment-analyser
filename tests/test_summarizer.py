from summarizer import rouge_n, summarize

TEXT = (
    "Information retrieval ranks documents by relevance to a query. "
    "The vector space model represents documents as TF-IDF weighted vectors. "
    "Cosine similarity compares vectors to rank documents for a query. "
    "Okapi BM25 improves ranking with term frequency saturation. "
    "PageRank scores nodes using the structure of a graph. "
    "TextRank applies PageRank to sentences to select a summary."
)


def test_all_methods_return_valid_selection():
    for m in ("hybrid", "textrank", "tfidf", "lsa"):
        r = summarize(TEXT, method=m, percent=50)
        assert len(r["pick"]) == 3 and r["pick"] == sorted(r["pick"])


def test_query_focus_prefers_matching_sentence():
    r = summarize(TEXT, method="tfidf", percent=17, query="PageRank graph")
    assert "PageRank" in r["S"][r["pick"][0]]


def test_too_short_returns_none():
    assert summarize("Too short.") is None


def test_rouge_identical():
    assert rouge_n("the cat sat", "the cat sat")["f1"] == 1.0
