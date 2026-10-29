"""Step 3: embed every abstract with the ten document representations.

Nine neural models (hrs_topics/embeddings.py lists each model's input format and pooling) plus a
TF-IDF baseline. A model whose output file already exists is skipped, so an interrupted run can
simply be restarted. Neural models need a GPU (CUDA or Apple MPS) to finish in reasonable time.

Input : data/corpus.csv
Output: data/embeddings/<model>.npy (float32, n_documents x dim), data/embeddings/tfidf.npz

Usage: python scripts/03_embed.py [--models mpnet,minilm,...] [--device cuda|mps|cpu] [--batch-size N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402
from hrs_topics.data import load_corpus  # noqa: E402
from hrs_topics.embeddings import SPECS, encode, fit_tfidf, input_texts  # noqa: E402


def default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(config.MODELS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()
    corpus = load_corpus()
    config.EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    for key in a.models.split(","):
        out = config.EMBEDDINGS / ("tfidf.npz" if key == "tfidf" else f"{key}.npy")
        if out.exists():
            print(f"{key}: exists, skipped")
            continue
        if key == "tfidf":
            X = fit_tfidf(corpus.embedding_text.astype(str).tolist())
            sp.save_npz(out, X)
        else:
            X = encode(SPECS[key], input_texts(SPECS[key], corpus), a.device or default_device(), a.batch_size)
            assert np.isfinite(X).all()
            np.save(out, X)
        print(f"{key}: {X.shape} -> {out}")


if __name__ == "__main__":
    main()
