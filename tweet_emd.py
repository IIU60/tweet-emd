"""Compute semantic distances between Twitter accounts using Earth Mover's Distance.

This script loads a dataset of tweets, embeds each tweet using a TF-IDF model
with latent semantic analysis, and measures the Earth Mover's Distance (EMD)
between the resulting embedding distributions of two accounts. The EMD is
obtained by solving a minimum-cost flow problem with NetworkX.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class AccountEmbeddings:
    """Container for the embeddings associated with a single account."""

    account: str
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("Embeddings must be provided as a 2D array.")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.vectors.shape[0]


def load_account_embeddings(
    dataset_path: Path | str,
    max_features: int = 5000,
    n_components: int = 256,
) -> Dict[str, AccountEmbeddings]:
    """Load tweets and compute embeddings grouped by account.

    Parameters
    ----------
    dataset_path:
        Path to the CSV file containing the tweet dataset.
    max_features:
        Maximum number of TF-IDF features to keep.
    n_components:
        Number of latent semantic components to retain with Truncated SVD.

    Returns
    -------
    Dict[str, AccountEmbeddings]
        Mapping from account name to its corresponding embeddings.
    """

    df = pd.read_csv(dataset_path)
    if "content" not in df or "author" not in df:
        raise ValueError("Dataset must contain 'content' and 'author' columns.")

    contents = df["content"].fillna("").tolist()
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    ).fit_transform(contents)

    if n_components is not None and n_components < tfidf.shape[1]:
        projector = TruncatedSVD(n_components=n_components, random_state=0)
        embedded = projector.fit_transform(tfidf)
    else:
        embedded = tfidf.toarray()

    embeddings = embedded.astype(np.float32)

    df = df.copy()
    df["embedding"] = list(embeddings)

    grouped: Dict[str, AccountEmbeddings] = {}
    for account, rows in df.groupby("author", sort=False):
        vectors = np.vstack(rows["embedding"].to_numpy())
        grouped[account] = AccountEmbeddings(account=account, vectors=vectors)

    return grouped


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two integers."""

    if a == 0 or b == 0:
        raise ValueError("Cannot compute LCM with zero.")
    return abs(a * b) // gcd(a, b)


def earth_movers_distance(
    left: AccountEmbeddings | np.ndarray,
    right: AccountEmbeddings | np.ndarray,
) -> float:
    """Compute the Earth Mover's Distance between two embedding sets.

    Parameters
    ----------
    left, right:
        Either ``AccountEmbeddings`` objects or numpy arrays of shape (n, d)
        and (m, d). The function supports distributions with different numbers
        of samples by evenly distributing the unit mass across tweets.

    Returns
    -------
    float
        The EMD between the two distributions.
    """

    left_vectors = left.vectors if isinstance(left, AccountEmbeddings) else np.asarray(left)
    right_vectors = right.vectors if isinstance(right, AccountEmbeddings) else np.asarray(right)

    if left_vectors.ndim != 2 or right_vectors.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")

    n_left, n_right = left_vectors.shape[0], right_vectors.shape[0]
    if n_left == 0 or n_right == 0:
        raise ValueError("Cannot compute EMD with empty embeddings.")

    graph = nx.DiGraph()
    total_mass = _lcm(n_left, n_right)
    left_mass = total_mass // n_left
    right_mass = total_mass // n_right

    # Add supply nodes for the left distribution.
    for idx in range(n_left):
        node = f"L{idx}"
        graph.add_node(node, demand=-left_mass)

    # Add demand nodes for the right distribution.
    for idx in range(n_right):
        node = f"R{idx}"
        graph.add_node(node, demand=right_mass)

    # Use an integer scaling factor so that min_cost_flow accepts the costs.
    cost_scale = 1000

    for i in range(n_left):
        for j in range(n_right):
            cost = float(np.linalg.norm(left_vectors[i] - right_vectors[j]))
            graph.add_edge(
                f"L{i}",
                f"R{j}",
                weight=int(round(cost * cost_scale)),
                capacity=left_mass,
            )

    flow_dict = nx.min_cost_flow(graph)

    total_cost = 0
    for i in range(n_left):
        for j in range(n_right):
            amount = flow_dict[f"L{i}"].get(f"R{j}", 0)
            if amount:
                total_cost += amount * graph[f"L{i}"][f"R{j}"]["weight"]

    emd_value = total_cost / (cost_scale * total_mass)
    return emd_value


def closest_account(
    target_account: str,
    embeddings: Dict[str, AccountEmbeddings],
) -> Tuple[str, float]:
    """Identify the account closest to ``target_account`` using EMD."""

    if target_account not in embeddings:
        raise KeyError(f"Unknown account: {target_account}")

    target_embeddings = embeddings[target_account]
    best_account: str | None = None
    best_distance: float | None = None

    for account, account_embeddings in embeddings.items():
        if account == target_account:
            continue

        distance = earth_movers_distance(target_embeddings, account_embeddings)
        if best_distance is None or distance < best_distance:
            best_account = account
            best_distance = distance

    if best_account is None or best_distance is None:
        raise RuntimeError("Failed to find a closest account.")

    return best_account, best_distance


def main(dataset_path: Path | str = "data/tweets_400.csv") -> None:  # pragma: no cover - CLI utility
    embeddings = load_account_embeddings(dataset_path)
    target = "ArianaGrande"
    account, distance = closest_account(target, embeddings)
    print(f"Closest account to {target}: {account} (EMD = {distance:.4f})")


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()
