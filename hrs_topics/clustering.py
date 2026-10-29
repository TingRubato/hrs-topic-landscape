"""UMAP reduction and HDBSCAN clustering with the project's fixed settings."""
import numpy as np

from . import config


def reduce(X, n_neighbors, min_dist, seed=config.SEED):
    """2-D UMAP coordinates. A fixed seed makes UMAP single-threaded and exactly reproducible."""
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed,
                        **config.UMAP_FIXED)
    return reducer.fit_transform(X).astype(np.float32)


def cluster(coords, min_cluster_size, min_samples, cluster_selection_epsilon):
    """HDBSCAN labels on UMAP coordinates; -1 marks outliers."""
    from sklearn.cluster import HDBSCAN
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon, copy=True,
                    **config.HDBSCAN_FIXED)
    return model.fit_predict(coords).astype(np.int16)


def fit(X, umap_id, hdbscan_id, seed=config.SEED):
    """UMAP then HDBSCAN for one grid setting, e.g. ("nn15_md0", "mcs50_ms5_cse0")."""
    u, h = config.umap_params(umap_id), config.hdbscan_params(hdbscan_id)
    coords = reduce(X, u["n_neighbors"], u["min_dist"], seed)
    return cluster(coords, h["min_cluster_size"], h["min_samples"], h["cluster_selection_epsilon"])
