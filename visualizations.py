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


def plot_emd_heatmap(embeddings: Dict[str, AccountEmbeddings], save_path: str | None = None, ax=None) -> None:
    """Display a heatmap of pairwise EMD distances between accounts."""

    accounts, matrix = _pairwise_emd(embeddings)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    im = ax.imshow(matrix, cmap="magma", origin="lower")

    ax.set_xticks(range(len(accounts)))
    ax.set_yticks(range(len(accounts)))
    ax.set_xticklabels(accounts, rotation=90, fontsize=7)
    ax.set_yticklabels(accounts, fontsize=7)
    ax.set_title("Pairwise EMD Heatmap", fontsize=9)

    if standalone:
        fig.colorbar(im, ax=ax, label="EMD (lower = more similar)")
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        else:
            plt.show()
        plt.close()
    else:
        plt.colorbar(im, ax=ax, label="EMD")


# ---------------------------------------------------------------------------
# Visual 2: Similarity network
# ---------------------------------------------------------------------------


def plot_similarity_network(
    embeddings: Dict[str, AccountEmbeddings],
    max_edges: int = 50,
    save_path: str | None = None,
    ax=None,
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

    # Use kamada_kawai_layout for better spacing, with increased iterations
    # If it fails, fall back to spring_layout with better parameters
    try:
        pos = nx.kamada_kawai_layout(graph, weight=None)
    except:
        pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)
    
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    
    # Normalize weights to 0-1 range and scale more dramatically
    if weights:
        min_w, max_w = min(weights), max(weights)
        if max_w > min_w:
            normalized_weights = [(w - min_w) / (max_w - min_w) for w in weights]
            # Scale from 0.5 to 8.0 for more visible differences
            edge_widths = [0.5 + w * 7.5 for w in normalized_weights]
        else:
            edge_widths = [2.0] * len(weights)
    else:
        edge_widths = [2.0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
        standalone = True
    else:
        fig = ax.figure
        standalone = False
    
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color="#8ecae6",
        edge_color="#219ebc",
        width=edge_widths,
        font_size=6 if not standalone else 8,
        node_size=200 if not standalone else 300,
    )
    ax.set_title("Similarity Network", fontsize=9)
    ax.axis("off")
    
    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved similarity network to {save_path}")
        else:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Visual 3: 2D projection of tweet embeddings
# ---------------------------------------------------------------------------


def plot_tweet_projection(
    embeddings: Dict[str, AccountEmbeddings],
    method: str = "pca",
    save_path: str | None = None,
    ax=None,
) -> None:
    """Project all tweet embeddings to 2D and color by account.

    Uses PCA for projection. The goal is simply to provide intuition: clusters indicate accounts
    with similar tweet semantics.
    """

    all_vectors: List[np.ndarray] = []
    labels: List[str] = []
    for account, bundle in embeddings.items():
        all_vectors.append(bundle.vectors)
        labels.extend([account] * len(bundle))

    stacked = np.vstack(all_vectors)

    # Use PCA for projection
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=42).fit_transform(stacked)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    # Match account names case-insensitively and handle variations
    def normalize_account_name(name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "")
    
    # Highlighted accounts with specific colors
    highlighted_accounts = {
        "arianagrande": {"color": (1.0, 0.0, 0.0, 0.8), "label": "Ariana Grande"},  # Red
        "twitter": {"color": (0.0, 0.4, 1.0, 0.8), "label": "Twitter (low EMD)"},  # Blue
        "barackobama": {"color": (1.0, 0.6, 0.0, 0.8), "label": "Barack Obama (high EMD)"},  # Orange
    }
    
    # Plot all accounts - highlighted ones with colors, others in grey
    for account in set(labels):
        acc_normalized = normalize_account_name(account)
        
        # Check if this account should be highlighted
        account_key = None
        for key in highlighted_accounts:
            if key in acc_normalized:
                account_key = key
                break
        
        mask = [label == account for label in labels]
        
        if account_key:
            # Highlighted account with color and label
            account_info = highlighted_accounts[account_key]
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=account_info["label"],
                color=account_info["color"],
                alpha=0.7,
                s=30,  # Larger points
            )
        else:
            # Other accounts in grey, no label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=(0.7, 0.7, 0.7, 0.2),  # Light grey with low alpha
                alpha=0.3,
                s=20,  # Smaller points
            )

    ax.set_title("Tweet embeddings projected to 2D")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", markerscale=2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved tweet projection to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Convenience CLI entry point
# ---------------------------------------------------------------------------


def main(dataset_path: str = "data/tweets_400.csv", max_edges: int = 50, save_dir: str | None = None) -> None:
    """Load embeddings then render all three visuals sequentially.

    Running ``python visualizations.py`` will compute embeddings for the supplied
    dataset (defaults to the bundled 400-tweet CSV) and pop up the heatmap,
    network, and 2D projection in order. The goal is to provide a single,
    copy-paste-friendly entry point for demos.
    
    If save_dir is provided, images will be saved instead of displayed.
    """

    embeddings = load_account_embeddings(dataset_path)

    # Plot 1: quick overview of all pairwise distances.
    heatmap_path = f"{save_dir}/emd_heatmap.png" if save_dir else None
    plot_emd_heatmap(embeddings, save_path=heatmap_path)

    # Plot 2: a graph view that highlights the strongest relationships.
    network_path = f"{save_dir}/similarity_network.png" if save_dir else None
    plot_similarity_network(embeddings, max_edges=max_edges, save_path=network_path)

    # Plot 3: point cloud showing tweet clusters per account.
    projection_path = f"{save_dir}/tweet_projection.png" if save_dir else None
    plot_tweet_projection(embeddings, save_path=projection_path)


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
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualization images (if not provided, images are displayed)",
    )

    args = parser.parse_args()
    main(dataset_path=args.dataset, max_edges=args.max_edges, save_dir=args.save_dir)
