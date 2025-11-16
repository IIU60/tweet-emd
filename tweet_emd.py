"""
Compute semantic distances between Twitter accounts using Earth Mover's Distance (EMD).

Changes vs the older version:
- Sentence-level embeddings (all-MiniLM-L6-v2) instead of TF-IDF.
- L2-normalization of every tweet vector.
- Cosine distance as ground cost: C_ij = 1 - x_i · y_j, with x,y unit vectors.
- Gentle tweet preprocessing: URL domains, split hashtags, mentions->@user, emojis->emoji_token.
- Exact EMD via NetworkX min_cost_flow with integer cost scaling.
- Utility to return top-5 most similar accounts.

The goal is to keep this file highly readable for beginners, with many comments
that explain *why* each step exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _split_hashtag(token: str) -> List[str]:
    """Split a hashtag token into smaller parts.

    The goal is not perfect linguistics—just a lightweight heuristic so
    ``#BestDayEver2024`` becomes ``["best", "day", "ever", "2024"]``. We look
    for transitions between lowercase/uppercase/number boundaries.
    """

    # Remove the leading '#', then split on transitions from lower->upper or
    # letter->digit. The regex keeps boundaries by inserting spaces we can split on.
    cleaned = token.lstrip("#")
    if not cleaned:
        return []

    parts = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])", " ", cleaned)
    return [p.lower() for p in parts.split() if p]


# Simple emoji pattern: matches surrogate pair ranges and common emoji blocks.
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F]|"  # emoticons
    r"[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
    r"[\U0001F680-\U0001F6FF]|"  # transport & map symbols
    r"[\U0001F700-\U0001F77F]|"  # alchemical symbols
    r"[\U0001F780-\U0001F7FF]|"  # geometric shapes extended
    r"[\U0001F800-\U0001F8FF]|"  # supplemental arrows-c
    r"[\U0001F900-\U0001F9FF]|"  # supplemental symbols & pictographs
    r"[\U0001FA00-\U0001FA6F]|"  # chess symbols, etc.
    r"[\u2600-\u26FF]|"           # miscellaneous symbols
    r"[\u2700-\u27BF]"            # dingbats
)


def preprocess_tweet(text: str) -> str:
    """Gently normalize a tweet while keeping most of its information.

    Steps (kept intentionally simple so the logic is easy to follow):
    - Lowercase everything so comparisons are case-insensitive.
    - Replace URLs with their domain (``url_domain_example.com``) so links still
      contribute semantic signal without huge unique tokens.
    - Replace mentions with a generic ``@user`` token to avoid overfitting on
      specific handles.
    - Expand hashtags: keep the raw hashtag token (``hashtag_bestday``) and also
      split camelCase/number boundaries to surface individual words.
    - Replace emojis with ``emoji_token`` to preserve their presence.
    - Collapse extra whitespace.
    """

    text = (text or "").strip().lower()

    # Replace URLs with their domain to keep some semantic cue.
    def _replace_url(match: re.Match[str]) -> str:
        url = match.group(0)
        # Extract the domain by stripping the protocol and path.
        domain_match = re.search(r"https?://([^/\s]+)", url)
        domain = domain_match.group(1) if domain_match else "link"
        # Remove a trailing slash if present.
        domain = domain.rstrip('/')
        return f"url_domain_{domain}"

    text = re.sub(r"https?://[^\s]+", _replace_url, text)

    # Replace mentions with a friendly placeholder.
    text = re.sub(r"@[A-Za-z0-9_]+", "@user", text)

    tokens: List[str] = []
    for raw_token in text.split():
        # Identify and replace emojis with a generic token. We keep only the
        # generic marker instead of removing them altogether.
        if _EMOJI_RE.search(raw_token):
            raw_token = _EMOJI_RE.sub(" emoji_token ", raw_token)

        # Handle hashtags: keep both the raw form and the split pieces.
        if raw_token.startswith("#"):
            tokens.append(f"hashtag_{raw_token.lstrip('#')}")
            tokens.extend(_split_hashtag(raw_token))
            continue

        tokens.append(raw_token)

    # Collapse repeated whitespace that may have appeared during replacements.
    cleaned = " ".join(token for token in tokens if token)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


@dataclass
class AccountEmbeddings:
    """Container holding an account name and its tweet embeddings."""

    account: str
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n_tweets, dim).")

    def __len__(self) -> int:  # pragma: no cover - trivial helper
        return self.vectors.shape[0]



def load_account_embeddings(dataset_path: Path | str) -> Dict[str, AccountEmbeddings]:
    """Load tweets from CSV, preprocess, embed, and group by account.

    Parameters
    ----------
    dataset_path:
        Path to a CSV file with ``author`` and ``content`` columns.

    Returns
    -------
    Dict[str, AccountEmbeddings]
        Mapping from account handle to its L2-normalized tweet embeddings.
    """

    df = pd.read_csv(dataset_path)
    if "content" not in df or "author" not in df:
        raise ValueError("Dataset must contain 'author' and 'content' columns.")

    # Apply lightweight normalization so the model sees consistent tokens.
    preprocessed = [preprocess_tweet(text) for text in df["content"].fillna("")]

    # Sentence-transformer handles sentence-level semantics far better than
    # bag-of-words TF-IDF, while remaining lightweight.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(preprocessed, convert_to_numpy=True, show_progress_bar=False)

    # Ensure each tweet vector has L2 norm = 1 so cosine distance is simply
    # ``1 - dot_product``.
    embeddings = normalize(embeddings, norm="l2")

    df = df.copy()
    df["embedding"] = list(embeddings.astype(np.float32))

    grouped: Dict[str, AccountEmbeddings] = {}
    for account, rows in df.groupby("author", sort=False):
        vectors = np.vstack(rows["embedding"].to_numpy())
        grouped[account] = AccountEmbeddings(account=account, vectors=vectors)

    return grouped


# ---------------------------------------------------------------------------
# Earth Mover's Distance via min-cost flow
# ---------------------------------------------------------------------------


def _lcm(a: int, b: int) -> int:
    """Least common multiple used to balance tweet masses."""

    if a == 0 or b == 0:
        raise ValueError("Cannot compute LCM with zero.")
    return abs(a * b) // gcd(a, b)


def earth_movers_distance(
    left: AccountEmbeddings | np.ndarray,
    right: AccountEmbeddings | np.ndarray,
) -> float:
    """Compute EMD between two sets of tweet embeddings using cosine cost.

    The cost between individual tweets is ``1 - x·y`` where both vectors are
    already L2-normalized. NetworkX expects *integer* costs, so we scale by
    ``cost_scale`` before solving, then divide to recover the floating value.
    Supplies/demands are balanced using the least common multiple so both sides
    carry equal total mass regardless of tweet counts.
    """

    left_vectors = left.vectors if isinstance(left, AccountEmbeddings) else np.asarray(left)
    right_vectors = right.vectors if isinstance(right, AccountEmbeddings) else np.asarray(right)

    if left_vectors.ndim != 2 or right_vectors.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")

    n_left, n_right = left_vectors.shape[0], right_vectors.shape[0]
    if n_left == 0 or n_right == 0:
        raise ValueError("Cannot compute EMD with empty embeddings.")

    # Compute cosine-based costs. Because vectors are normalized, dot products
    # live in [-1, 1], so ``1 - dot`` lives in [0, 2].
    cost_matrix = 1.0 - np.clip(left_vectors @ right_vectors.T, -1.0, 1.0)

    # NetworkX requires integer weights. Scaling by 10_000 preserves four
    # decimal places while keeping numbers small enough for exact solvers.
    cost_scale = 10_000
    scaled_costs = np.round(cost_matrix * cost_scale).astype(int)

    total_mass = _lcm(n_left, n_right)
    left_mass = total_mass // n_left
    right_mass = total_mass // n_right

    graph = nx.DiGraph()

    # Supply nodes (left tweets) provide mass.
    for idx in range(n_left):
        graph.add_node(f"L{idx}", demand=-left_mass)

    # Demand nodes (right tweets) consume mass.
    for idx in range(n_right):
        graph.add_node(f"R{idx}", demand=right_mass)

    # Each left tweet can send its full mass to any right tweet.
    for i in range(n_left):
        for j in range(n_right):
            graph.add_edge(
                f"L{i}",
                f"R{j}",
                weight=int(scaled_costs[i, j]),
                capacity=left_mass,
            )

    flow = nx.min_cost_flow(graph)

    total_cost = 0
    for i in range(n_left):
        for j in range(n_right):
            shipped = flow[f"L{i}"].get(f"R{j}", 0)
            if shipped:
                total_cost += shipped * graph[f"L{i}"][f"R{j}"]["weight"]

    # Rescale to the original floating-space EMD.
    emd_value = total_cost / (cost_scale * total_mass)
    return emd_value


# ---------------------------------------------------------------------------
# Top-k neighbors and CLI
# ---------------------------------------------------------------------------


def top_k_similar_accounts(
    target: str,
    embeddings: Dict[str, AccountEmbeddings],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k accounts closest to ``target`` by EMD.

    The target account is excluded from the results. Distances are sorted
    ascending (smaller EMD means more similar semantics).
    """

    if target not in embeddings:
        raise KeyError(f"Unknown target account: {target}")

    target_embeddings = embeddings[target]
    distances: List[Tuple[str, float]] = []
    for account, account_embeddings in embeddings.items():
        if account == target:
            continue
        distance = earth_movers_distance(target_embeddings, account_embeddings)
        distances.append((account, distance))

    distances.sort(key=lambda pair: pair[1])
    return distances[: max(0, min(k, len(distances)))]


def main(dataset_path: Path | str = "data/tweets_400.csv") -> None:  # pragma: no cover - CLI helper
    """Small CLI demo that prints a sanity self-distance and the top-5 neighbors."""

    embeddings = load_account_embeddings(dataset_path)

    # Pick the first account as a quick demo target.
    target = next(iter(embeddings))

    self_distance = earth_movers_distance(embeddings[target], embeddings[target])
    print(f"Self EMD for {target}: {self_distance:.6f} (should be ~0)")

    print(f"Top similar accounts to {target}:")
    for account, distance in top_k_similar_accounts(target, embeddings, k=5):
        print(f"  - {account}: EMD = {distance:.4f}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/tweets_400.csv",
        help="Path to the tweets CSV (default: data/tweets_400.csv)",
    )
    args = parser.parse_args()

    main(args.dataset)
