from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def export_full_memmap_gpu(
    xn: np.ndarray,
    cn: np.ndarray,
    out_path: Path,
    batch: int,
    out_dtype: str = "float32",
    device: str = "cuda:0",
) -> None:
    """
    Requires torch + CUDA.
    Computes dist = 1 - (xb @ cn.T) on GPU, writes to CPU memmap.

    xn, cn should be float32 normalized on CPU before calling.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("GPU export requires torch. Install requirements-gpu.txt") from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --backend cpu or install CUDA-enabled PyTorch.")

    N, D = xn.shape
    K = cn.shape[0]
    out_dtype_np = np.dtype(out_dtype)
    out = np.memmap(str(out_path), mode="w+", dtype=out_dtype_np, shape=(N, K))

    # Move centroids to GPU once
    cn_t = torch.from_numpy(cn).to(device=device, dtype=torch.float32).t().contiguous()  # (D,K)

    for i in range(0, N, batch):
        xb = xn[i:i+batch]  # CPU
        xb_gpu = torch.from_numpy(xb).to(device=device, dtype=torch.float32, non_blocking=True)
        sim = xb_gpu @ cn_t                 # (B,K) on GPU
        dist = 1.0 - sim
        dist_cpu = dist.to("cpu").numpy()   # float32
        out[i:i+xb.shape[0]] = dist_cpu.astype(out_dtype_np, copy=False)

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
        "backend": "gpu",
        "device": device,
    }
    (out_path.with_suffix(out_path.suffix + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")