# Tweet EMD: Semantic Distance Analysis

Compute semantic distances between Twitter accounts using Earth Mover's Distance (EMD).

## Overview

This project implements a semantic similarity analysis system for Twitter accounts using Earth Mover's Distance (EMD). It processes tweets, generates embeddings, and computes the EMD between account tweet distributions to identify similar accounts.

## Features

- **Sentence-level embeddings**: Uses `all-MiniLM-L6-v2` sentence transformer instead of TF-IDF for better semantic understanding
- **L2-normalization**: All tweet vectors are normalized for consistent cosine distance calculations
- **Cosine distance**: Ground cost metric `C_ij = 1 - x_i · y_j` for unit vectors
- **Tweet preprocessing**: 
  - URL domain extraction
  - Hashtag splitting (camelCase detection)
  - Mention normalization (`@user`)
  - Emoji tokenization
- **Exact EMD computation**: Uses NetworkX min-cost flow with integer cost scaling
- **Top-k similar accounts**: Utility function to find the most similar accounts

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `sentence-transformers`
- `networkx`
- `numpy`
- `pandas`
- `scikit-learn`
- `pytest` (for testing)

## Usage

### Basic Usage

```python
from tweet_emd import load_account_embeddings, top_k_similar_accounts

# Load embeddings from CSV
embeddings = load_account_embeddings("tweets_400.csv")

# Find top 5 similar accounts
target = "SomeAccount"
similar = top_k_similar_accounts(target, embeddings, k=5)
for account, distance in similar:
    print(f"{account}: EMD = {distance:.4f}")
```

### Command Line

```bash
python tweet_emd.py --dataset tweets_400.csv
```

## Data Format

The input CSV file should have the following columns:
- `author`: Twitter account handle
- `content`: Tweet text content

## Project Structure

- `tweet_emd.py`: Main implementation with sentence embeddings
- `test_tweet_emd.py`: Test suite
- `tweets_400.csv`: Sample dataset

## History

This project evolved from a TF-IDF based implementation to the current sentence embedding approach. The git history preserves this evolution, showing the transition from the earlier version.

## License

[Add your license here]

