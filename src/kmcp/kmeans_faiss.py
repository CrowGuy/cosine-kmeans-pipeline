from __future__ import annotations
import numpy as np
import faiss

def train_faiss_kmeans(
    x_train: np.ndarray,
    k: int,
    niter: int = 25,
    nredo: int = 1,
    seed: int = 123,
    max_points_per_centroid: int = 5000,
    use_gpu: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    if x_train.dtype != np.float32:
        raise ValueError("FAISS KMeans expects float32 input")

    d = x_train.shape[1]
    km = faiss.Kmeans(
        d=d,
        k=k,
        niter=niter,
        nredo=nredo,
        verbose=verbose,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        gpu=use_gpu,
    )
    km.train(x_train)
    centroids = faiss.vector_to_array(km.centroids).reshape(k, d).astype(np.float32, copy=False)
    obj_last = float(km.obj[-1]) if len(km.obj) else float("nan")
    return centroids, obj_last

def assign_nearest_by_ip(xn: np.ndarray, cn: np.ndarray, topk: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Use inner product search. Assumes xn and cn are L2-normalized for cosine usage.
    Returns (sim, idx):
      sim: (N, topk) float32
      idx: (N, topk) int64
    """
    d = xn.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(cn)
    sim, idx = index.search(xn, topk)
    return sim, idx