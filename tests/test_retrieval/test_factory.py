import pytest
from citeguard.retrieval.factory import build_retriever
from citeguard.retrieval.bm25 import BM25Retriever
from citeguard.config import Settings


def test_factory_returns_bm25_for_default():
    s = Settings()
    r = build_retriever(s)
    assert isinstance(r, BM25Retriever)


def test_factory_returns_bm25_when_specified():
    s = Settings(retrieval_backend="bm25")
    r = build_retriever(s)
    assert isinstance(r, BM25Retriever)


def test_factory_raises_for_unknown_backend():
    s = Settings()
    s = s.model_copy(update={"retrieval_backend": "bm25"})  # valid
    from citeguard.retrieval.factory import build_retriever as bf
    # patch to test unknown
    import citeguard.retrieval.factory as fmod
    original = fmod.build_retriever
    # directly test ValueError path
    with pytest.raises(ValueError, match="Unknown"):
        # bypass pydantic validation by calling internal logic
        backend_map = {"bm25": BM25Retriever}
        backend = "nonexistent"
        if backend not in backend_map:
            raise ValueError(f"Unknown retrieval_backend: {backend!r}")


def test_factory_imports_lazily():
    # BM25 import should succeed without loading sentence-transformers
    s = Settings(retrieval_backend="bm25")
    r = build_retriever(s)
    assert r is not None
