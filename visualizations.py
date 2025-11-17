"""Simple plotting helpers to explain the tweet-EMD pipeline.

All functions accept the dictionary returned by ``load_account_embeddings`` and
produce straightforward matplotlib figures suitable for a quick classroom demo.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tweet_emd import (
    AccountEmbeddings,
    earth_movers_distance,
    load_account_embeddings,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _pairwise_emd(embeddings: Dict[str, AccountEmbeddings]) -> Tuple[List[str], np.ndarray]:
    """Compute a small pairwise EMD matrix for plotting purposes."""

    accounts = list(embeddings)
    n = len(accounts)
    matrix = np.zeros((n, n), dtype=float)

    for i, ai in enumerate(accounts):
        for j in range(i + 1, n):
            aj = accounts[j]
            dist = earth_movers_distance(embeddings[ai], embeddings[aj])
            matrix[i, j] = matrix[j, i] = dist

    return accounts, matrix


# ---------------------------------------------------------------------------
# Visual 1: Heatmap of pairwise EMD distances
# ---------------------------------------------------------------------------


def plot_emd_heatmap(embeddings: Dict[str, AccountEmbeddings]) -> None:
    """Display a heatmap of pairwise EMD distances between accounts."""

    accounts, matrix = _pairwise_emd(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="magma", origin="lower")

    ax.set_xticks(range(len(accounts)))
    ax.set_yticks(range(len(accounts)))
    ax.set_xticklabels(accounts, rotation=90)
    ax.set_yticklabels(accounts)
    ax.set_title("Pairwise Earth Mover's Distance between accounts")

    fig.colorbar(im, ax=ax, label="EMD (lower = more similar)")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visual 2: Similarity network
# ---------------------------------------------------------------------------


def plot_similarity_network(
    embeddings: Dict[str, AccountEmbeddings],
    max_edges: int = 50,
) -> None:
    """Draw a graph where edges connect semantically similar accounts.

    Similarity is defined as ``1 / (1 + emd)`` so that smaller distances produce
    larger similarity scores. Only the strongest ``max_edges`` edges are drawn to
    keep the picture readable.
    """

    accounts, matrix = _pairwise_emd(embeddings)
    similarities: List[Tuple[str, str, float]] = []

    for i, ai in enumerate(accounts):
        for j in range(i + 1, len(accounts)):
            aj = accounts[j]
            emd = matrix[i, j]
            similarity = 1.0 / (1.0 + emd)
            similarities.append((ai, aj, similarity))

    # Keep only the strongest edges.
    similarities.sort(key=lambda item: item[2], reverse=True)
    edges_to_draw = similarities[:max_edges]

    graph = nx.Graph()
    for account in accounts:
        graph.add_node(account)

    for a, b, sim in edges_to_draw:
        graph.add_edge(a, b, weight=sim)

    pos = nx.spring_layout(graph, seed=42)
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        node_color="#8ecae6",
        edge_color="#219ebc",
        width=[w * 4 for w in weights],
        font_size=9,
    )
    plt.title("Account similarity network (thicker = closer)")
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Visual 3: 2D projection of tweet embeddings
# ---------------------------------------------------------------------------


def plot_tweet_projection(
    embeddings: Dict[str, AccountEmbeddings],
    method: str = "umap",
) -> None:
    """Project all tweet embeddings to 2D and color by account.

    Uses UMAP when available (nice non-linear projection), otherwise falls back
    to PCA. The goal is simply to provide intuition: clusters indicate accounts
    with similar tweet semantics.
    """

    all_vectors: List[np.ndarray] = []
    labels: List[str] = []
    for account, bundle in embeddings.items():
        all_vectors.append(bundle.vectors)
        labels.extend([account] * len(bundle))

    stacked = np.vstack(all_vectors)

    # Try UMAP first; if unavailable, drop back to PCA for reliability.
    coords: np.ndarray
    if method.lower() == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(random_state=42)
            coords = reducer.fit_transform(stacked)
        except Exception:
            from sklearn.decomposition import PCA

            coords = PCA(n_components=2, random_state=42).fit_transform(stacked)
    else:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(stacked)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Assign a consistent color to each account using matplotlib's tab20 palette.
    palette = plt.get_cmap("tab20")
    account_to_color = {acc: palette(i % 20) for i, acc in enumerate(set(labels))}

    for account in account_to_color:
        mask = [label == account for label in labels]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=account,
            color=account_to_color[account],
            alpha=0.6,
            s=15,
        )

    ax.set_title("Tweet embeddings projected to 2D")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", markerscale=2)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Convenience CLI entry point
# ---------------------------------------------------------------------------


def main(dataset_path: str = "data/tweets_400.csv", max_edges: int = 50) -> None:
    """Load embeddings then render all three visuals sequentially.

    Running ``python visualizations.py`` will compute embeddings for the supplied
    dataset (defaults to the bundled 400-tweet CSV) and pop up the heatmap,
    network, and 2D projection in order. The goal is to provide a single,
    copy-paste-friendly entry point for demos.
    """

    embeddings = load_account_embeddings(dataset_path)

    # Plot 1: quick overview of all pairwise distances.
    plot_emd_heatmap(embeddings)

    # Plot 2: a graph view that highlights the strongest relationships.
    plot_similarity_network(embeddings, max_edges=max_edges)

    # Plot 3: point cloud showing tweet clusters per account.
    plot_tweet_projection(embeddings)


if __name__ == "__main__":
    # Keep the CLI minimal and friendly: optional dataset and max_edges arguments
    # can be provided positionally when invoking the script.
    import argparse

    parser = argparse.ArgumentParser(description="Run all tweet-EMD visuals.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/tweets_400.csv",
        help="Path to the tweets CSV (default: data/tweets_400.csv)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=50,
        help="Maximum number of edges to draw in the similarity network (default: 50)",
    )

    args = parser.parse_args()
    main(dataset_path=args.dataset, max_edges=args.max_edges)
