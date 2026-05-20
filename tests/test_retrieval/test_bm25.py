import pytest
from pathlib import Path
from citeguard.retrieval.base import RetrieverBase
from citeguard.retrieval.bm25 import BM25Retriever
from citeguard.models import ChunkRecord


def make_chunks(texts: list[str]) -> list[ChunkRecord]:
    return [
        ChunkRecord(pdf_path="test.pdf", chunk_id=f"c{i:03d}", text=t, page=1)
        for i, t in enumerate(texts)
    ]


def test_bm25_is_retriever_base():
    assert issubclass(BM25Retriever, RetrieverBase)


def test_query_returns_ranked_results():
    r = BM25Retriever()
    chunks = make_chunks([
        "neural networks achieve high accuracy on image tasks",
        "decision trees are interpretable models",
        "accuracy of neural network classifiers on benchmarks",
    ])
    r.index(chunks)
    results = r.query("neural network accuracy", top_k=2)
    assert len(results) == 2
    assert "neural" in results[0].text.lower()


def test_query_scores_are_non_negative():
    r = BM25Retriever()
    r.index(make_chunks(["relevant text about transformers", "unrelated weather report"]))
    results = r.query("transformers", top_k=2)
    assert all(c.score >= 0 for c in results)
    assert results[0].score >= results[1].score


def test_query_empty_index_returns_empty():
    r = BM25Retriever()
    r.index([])
    results = r.query("anything", top_k=5)
    assert results == []


def test_save_and_load_roundtrip(tmp_path):
    r = BM25Retriever()
    chunks = make_chunks(["save and load test document", "another chunk for testing"])
    r.index(chunks)
    path = tmp_path / "index.pkl"
    r.save(path)
    r2 = BM25Retriever.load(path)
    results = r2.query("save load", top_k=1)
    assert len(results) == 1
    assert "save" in results[0].text


def test_top_k_respected():
    r = BM25Retriever()
    r.index(make_chunks([f"chunk number {i} with important content" for i in range(20)]))
    results = r.query("chunk content", top_k=5)
    assert len(results) == 5


def test_query_does_not_exceed_available_chunks():
    r = BM25Retriever()
    r.index(make_chunks(["only one chunk"]))
    results = r.query("one chunk", top_k=10)
    assert len(results) == 1


def test_chunk_scores_set_on_results():
    r = BM25Retriever()
    r.index(make_chunks(["highly relevant content about bert", "unrelated data"]))
    results = r.query("bert", top_k=2)
    assert results[0].score > 0
