from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

DistanceDType = Literal["float16", "float32"]
ExportBackend = Literal["cpu", "gpu"]
InputFormat = Literal["npy", "raw_mmap"]

@dataclass(frozen=True)
class ArraySpec:
    path: str
    fmt: InputFormat = "npy"
    mmap: bool = False
    shape: Optional[Tuple[int, int]] = None
    dtype: str = "float32"