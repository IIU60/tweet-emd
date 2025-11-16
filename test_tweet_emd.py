"""Pytest sanity checks for the tweet EMD pipeline.

The tests monkey-patch the SentenceTransformer model to avoid heavy downloads
and to keep outputs deterministic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

import tweet_emd as te


class _StubModel:
    """Lightweight stand-in for SentenceTransformer used in tests.

    It returns deterministic unit vectors based on a fixed random seed so that
    tests are stable across runs and machines.
    """

    def __init__(self) -> None:
        self.rng = np.random.default_rng(seed=0)

    def encode(self, texts: List[str], convert_to_numpy: bool = True, show_progress_bar: bool = False):
        # Generate a random vector per text and normalize so norms are ~1.
        vectors = self.rng.standard_normal((len(texts), 8))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, te.AccountEmbeddings]:
    """Create a small CSV and patch the embedding model for speed."""

    data = pd.DataFrame(
        {
            "author": ["A", "A", "B", "B", "C"],
            "content": [
                "Check out https://example.com #Fun 😊",
                "Hello @someone! #FunTime",
                "Different topic entirely",
                "Another thought",
                "A different account",
            ],
        }
    )
    csv_path = tmp_path / "tweets.csv"
    data.to_csv(csv_path, index=False)

    monkeypatch.setattr(te, "SentenceTransformer", lambda *_, **__: _StubModel())
    embeddings = te.load_account_embeddings(csv_path)
    return embeddings


def test_embeddings_are_normalized(tiny_dataset: Dict[str, te.AccountEmbeddings]) -> None:
    """Every tweet vector should be unit-length after preprocessing."""

    for bundle in tiny_dataset.values():
        norms = np.linalg.norm(bundle.vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


def test_self_distance_is_zero(tiny_dataset: Dict[str, te.AccountEmbeddings]) -> None:
    """EMD of an account with itself should be ~0 (integer scaling may leave tiny noise)."""

    for account, bundle in tiny_dataset.items():
        assert te.earth_movers_distance(bundle, bundle) < 1e-6


def test_symmetry(tiny_dataset: Dict[str, te.AccountEmbeddings]) -> None:
    """EMD should be symmetric."""

    accounts = list(tiny_dataset)
    left = tiny_dataset[accounts[0]]
    right = tiny_dataset[accounts[1]]
    assert te.earth_movers_distance(left, right) == pytest.approx(
        te.earth_movers_distance(right, left),
        abs=1e-9,
    )


def test_top_k_returns_sorted_results(tiny_dataset: Dict[str, te.AccountEmbeddings]) -> None:
    """Top-k should exclude the target, be sorted, and have length up to k."""

    target = next(iter(tiny_dataset))
    results = te.top_k_similar_accounts(target, tiny_dataset, k=5)

    # Target should not appear in the list.
    assert all(account != target for account, _ in results)
    # Sorted by distance ascending.
    distances = [d for _, d in results]
    assert distances == sorted(distances)
    # Length is min(k, number of other accounts).
    assert len(results) == min(5, len(tiny_dataset) - 1)


def test_cosine_cost_bounds() -> None:
    """Cosine-based cost should stay within [0, 2]."""

    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    c = np.array([[-1.0, 0.0]])

    # Identical directions yield cost ~0.
    assert te.earth_movers_distance(a, b) == pytest.approx(0.0, abs=1e-9)
    # Opposite directions yield cost ~2.
    assert te.earth_movers_distance(a, c) == pytest.approx(2.0, rel=1e-6)
