from __future__ import annotations
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score

from .normalize import random_sample, l2_normalize_inplace
from .kmeans_faiss import train_faiss_kmeans, assign_nearest_by_ip

def scan_k(
    X: np.ndarray,
    k_list: list[int],
    train_sample: int,
    eval_sample: int,
    silhouette_eval: int,
    seed: int,
    use_gpu: bool,
    niter: int,
    max_points_per_centroid: int,
    verbose: bool = False,
) -> list[dict]:
    Xt = random_sample(X, train_sample, seed=seed).astype(np.float32, copy=False)
    Xe = random_sample(X, eval_sample, seed=seed+1).astype(np.float32, copy=False)
    l2_normalize_inplace(Xt)
    l2_normalize_inplace(Xe)

    Xs = random_sample(Xe, silhouette_eval, seed=seed+2)
    results = []

    for k in k_list:
        C, obj = train_faiss_kmeans(
            Xt, k=k, niter=niter, nredo=1, seed=seed,
            max_points_per_centroid=max_points_per_centroid,
            use_gpu=use_gpu, verbose=False
        )
        l2_normalize_inplace(C)
        sim, idx = assign_nearest_by_ip(Xe, C, topk=1)
        labels_e = idx.reshape(-1)

        # silhouette 用 cosine（因為我們 normalize 了）
        labels_s = labels_e[:Xs.shape[0]]
        sil = float(silhouette_score(Xs, labels_s, metric="cosine"))

        rec = {"k": int(k), "obj": float(obj), "silhouette_cosine": sil}
        results.append(rec)
        if verbose:
            print(rec)

    return results

def stability_ari(
    X: np.ndarray,
    k: int,
    train_sample: int,
    eval_sample: int,
    repeats: int,
    seed: int,
    use_gpu: bool,
    niter: int,
    max_points_per_centroid: int,
    verbose: bool = False,
) -> dict:
    Xe = random_sample(X, eval_sample, seed=seed+999).astype(np.float32, copy=False)
    l2_normalize_inplace(Xe)

    labels_runs = []
    for r in range(repeats):
        Xt = random_sample(X, train_sample, seed=seed+r).astype(np.float32, copy=False)
        l2_normalize_inplace(Xt)

        C, _ = train_faiss_kmeans(
            Xt, k=k, niter=niter, nredo=1, seed=seed+r,
            max_points_per_centroid=max_points_per_centroid,
            use_gpu=use_gpu, verbose=False
        )
        l2_normalize_inplace(C)
        _, idx = assign_nearest_by_ip(Xe, C, topk=1)
        labels_runs.append(idx.reshape(-1))

        if verbose:
            print(f"[k={k}] repeat {r+1}/{repeats}")

    aris = []
    for i in range(repeats):
        for j in range(i+1, repeats):
            aris.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))

    return {
        "k": int(k),
        "stability_ari_mean": float(np.mean(aris)),
        "stability_ari_std": float(np.std(aris)),
        "repeats": int(repeats),
        "eval_sample": int(eval_sample),
        "train_sample": int(train_sample),
    }
