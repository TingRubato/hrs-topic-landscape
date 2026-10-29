"""Loading the corpus, embeddings and grid outputs written by the pipeline steps."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config


def load_corpus():
    """corpus.csv; its row order is the row order of every embedding and label array."""
    return pd.read_csv(config.CORPUS)


def load_embedding(model):
    """Dense float32 matrix, or a sparse CSR matrix for TF-IDF."""
    if model == "tfidf":
        return sp.load_npz(config.EMBEDDINGS / "tfidf.npz")
    return np.load(config.EMBEDDINGS / f"{model}.npy")


def load_features(model):
    """Input to UMAP: the embedding itself, or for TF-IDF its 300-component truncated SVD."""
    X = load_embedding(model)
    if sp.issparse(X):
        from sklearn.decomposition import TruncatedSVD
        k = min(config.TFIDF_SVD_COMPONENTS, X.shape[1] - 1)
        return TruncatedSVD(n_components=k, random_state=config.SEED).fit_transform(X)
    return X.astype(np.float32)


def umap_path(model, umap_id):
    return config.GRID / "umap" / f"{model}__{umap_id}.npy"


def labels_path(model, umap_id, hdbscan_id):
    return config.GRID / "labels" / f"{model}__{umap_id}__{hdbscan_id}.npy"


def load_labels(model, umap_id, hdbscan_id):
    """HDBSCAN topic labels per document; -1 = outlier."""
    return np.load(labels_path(model, umap_id, hdbscan_id)).astype(int)
