from __future__ import annotations
import numpy as np

def l2_norm_stats(x: np.ndarray) -> dict:
    norms = np.linalg.norm(x.astype(np.float32, copy=False), axis=1)
    return {
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
        "std": float(norms.std()),
    }

def l2_normalize_inplace(x: np.ndarray, eps: float = 1e-12) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x /= (norms + eps)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)

def random_sample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    n = min(n, x.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]