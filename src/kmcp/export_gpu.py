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

    torch_device = torch.device(device)

    with torch.no_grad():
        # centroids^T on GPU once: (D,K)
        cn_t = torch.from_numpy(cn).to(device=torch_device, dtype=torch.float32).t().contiguous()

        for i in range(0, N, batch):
            xb = xn[i:i+batch]  # CPU float32

            # pinned memory -> faster H2D
            xb_cpu = torch.from_numpy(xb).pin_memory()
            xb_gpu = xb_cpu.to(device=torch_device, dtype=torch.float32, non_blocking=True)

            sim = xb_gpu @ cn_t
            dist = 1.0 - sim

            dist_cpu = dist.to("cpu").numpy()  # float32
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