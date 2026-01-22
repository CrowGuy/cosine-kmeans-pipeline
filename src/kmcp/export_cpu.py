from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .normalize import l2_normalize

def export_topm_cpu(
    xn: np.ndarray, cn: np.ndarray, m: int, out_dir: Path, batch: int
) -> None:
    # cosine sim = dot(xn, cn)
    N, D = xn.shape
    K = cn.shape[0]
    idx_out = np.empty((N, m), dtype=np.int32)
    dist_out = np.empty((N, m), dtype=np.float32)

    cn_t = cn.T  # (D,K)
    for i in range(0, N, batch):
        xb = xn[i:i+batch]                 # (B,D)
        sim = xb @ cn_t                    # (B,K)
        # top-m highest sim => smallest distance
        top_idx = np.argpartition(-sim, kth=m-1, axis=1)[:, :m]
        top_sim = np.take_along_axis(sim, top_idx, axis=1)

        # sort each row by similarity desc (optional but nicer)
        order = np.argsort(-top_sim, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)

        idx_out[i:i+xb.shape[0]] = top_idx.astype(np.int32, copy=False)
        dist_out[i:i+xb.shape[0]] = (1.0 - top_sim).astype(np.float32, copy=False)

    np.save(out_dir / "topm_idx.npy", idx_out)
    np.save(out_dir / "topm_dist.npy", dist_out)

    meta = {
        "type": "topm",
        "N": int(N), "K": int(K), "D": int(D),
        "m": int(m),
        "distance": "cosine_distance = 1 - dot(normalize(x), normalize(c))",
        "idx_file": "topm_idx.npy",
        "dist_file": "topm_dist.npy",
        "dist_dtype": "float32",
        "idx_dtype": "int32",
    }
    (out_dir / "topm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def export_full_memmap_cpu(
    xn: np.ndarray,
    cn: np.ndarray,
    out_path: Path,
    batch: int,
    out_dtype: str = "float32",
) -> None:
    N, D = xn.shape
    K = cn.shape[0]
    out_dtype_np = np.dtype(out_dtype)
    out = np.memmap(str(out_path), mode="w+", dtype=out_dtype_np, shape=(N, K))

    cn_t = cn.T  # (D,K)
    for i in range(0, N, batch):
        xb = xn[i:i+batch]
        sim = xb @ cn_t
        dist = 1.0 - sim
        out[i:i+xb.shape[0]] = dist.astype(out_dtype_np, copy=False)

    out.flush()
    meta = {
        "type": "full",
        "N": int(N), "K": int(K), "D": int(D),
        "distance": "cosine_distance = 1 - dot(normalize(x), normalize(c))",
        "file": out_path.name,
        "format": "raw_memmap",
        "dtype": out_dtype,
        "shape": [int(N), int(K)],
        "order": "C",
    }
    (out_path.with_suffix(out_path.suffix + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")