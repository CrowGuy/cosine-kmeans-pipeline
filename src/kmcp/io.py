from __future__ import annotations
import numpy as np
from .types import ArraySpec

def load_array(spec: ArraySpec) -> np.ndarray:
    if spec.fmt == "raw_mmap":
        if spec.shape is None:
            raise ValueError("raw_mmap requires shape=(N,D)")
        return np.memmap(spec.path, mode="r", dtype=np.dtype(spec.dtype), shape=spec.shape)

    # npy
    if spec.mmap:
        return np.load(spec.path, mmap_mode="r")
    return np.load(spec.path)

def ensure_float32_contiguous(x: np.ndarray) -> np.ndarray:
    # for FAISS training and fast GEMM
    if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
        return np.array(x, dtype=np.float32, copy=True, order="C")
    # if it's memmap read-only, still ok for training if we don't normalize in-place
    return x