from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer
import yaml
import numpy as np
from tqdm import tqdm

from .types import ArraySpec, ExportBackend
from .io import load_array, ensure_float32_contiguous
from .normalize import random_sample, l2_norm_stats, l2_normalize_inplace
from .kmeans_faiss import train_faiss_kmeans
from .export_cpu import export_topm_cpu, export_full_memmap_cpu
from .export_gpu import export_full_memmap_gpu
from .select_k import scan_k, stability_ari

app = typer.Typer(add_completion=False)

def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _load_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

def _array_spec_from_cfg(cfg: dict) -> ArraySpec:
    data = cfg.get("data", {})
    fmt = data.get("format", "npy")
    mmap = bool(data.get("mmap", False))
    shape = data.get("shape")
    shape_t = tuple(shape) if shape else None
    dtype = data.get("dtype", "float32")
    return ArraySpec(
        path=data["input"],
        fmt=fmt,
        mmap=mmap,
        shape=shape_t,
        dtype=dtype,
    )

@app.command()
def check(
    config: Optional[str] = typer.Option(None, "--config"),
    out_dir: str = typer.Option("runs/check", "--out-dir"),
    sample_size: int = typer.Option(20000, "--sample-size"),
    seed: int = typer.Option(123, "--seed"),
):
    """Check L2 norm distribution on a random sample."""
    cfg = _load_config(config)
    spec = _array_spec_from_cfg(cfg)
    X = load_array(spec)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    Xs = random_sample(np.array(X, copy=False), sample_size, seed=seed)
    stats = l2_norm_stats(Xs)
    meta = {"sample_size": int(min(sample_size, X.shape[0])), "stats": stats}
    _save_json(out / "norm_check.json", meta)
    typer.echo(f"Saved: {out / 'norm_check.json'}")
    typer.echo(f"Stats: {stats}")

@app.command("select-k")
def select_k_cmd(
    config: Optional[str] = typer.Option(None, "--config"),
    out_dir: str = typer.Option("runs/select_k", "--out-dir"),
    seed: int = typer.Option(123, "--seed"),
    verbose: bool = typer.Option(False, "--verbose"),
):
    """Run k scan and stability ARI (based on config)."""
    cfg = _load_config(config)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    spec = _array_spec_from_cfg(cfg)
    X = load_array(spec)
    X32 = ensure_float32_contiguous(X)  # ok if it copies; selection uses samples anyway

    ks = cfg["k_select"]["k_list"]
    use_gpu = bool(cfg.get("kmeans", {}).get("use_gpu", False))
    niter = int(cfg.get("kmeans", {}).get("niter", 25))
    max_ppc = int(cfg.get("kmeans", {}).get("max_points_per_centroid", 5000))

    scan_cfg = cfg["k_select"]
    scan_res = scan_k(
        X32,
        k_list=ks,
        train_sample=int(scan_cfg.get("train_sample", 200000)),
        eval_sample=int(scan_cfg.get("eval_sample", 20000)),
        silhouette_eval=int(scan_cfg.get("silhouette_eval", 10000)),
        seed=seed,
        use_gpu=use_gpu,
        niter=niter,
        max_points_per_centroid=max_ppc,
        verbose=verbose,
    )
    _save_json(out / "k_scan.json", {"results": scan_res})

    # pick top 3 by silhouette for stability, plus keep any user-preferred k if present
    top3 = sorted(scan_res, key=lambda r: r["silhouette_cosine"], reverse=True)[:3]
    cand = [r["k"] for r in top3]

    stab_cfg = scan_cfg.get("stability", {})
    stab_enabled = bool(stab_cfg.get("enabled", True))
    stabs = []
    if stab_enabled:
        for k in cand:
            stabs.append(stability_ari(
                X32, k=k,
                train_sample=int(stab_cfg.get("train_sample", 200000)),
                eval_sample=int(stab_cfg.get("eval_sample", 50000)),
                repeats=int(stab_cfg.get("repeats", 6)),
                seed=seed,
                use_gpu=use_gpu,
                niter=niter,
                max_points_per_centroid=max_ppc,
                verbose=verbose,
            ))
        _save_json(out / "k_stability.json", {"results": stabs})

    # recommend: highest stability mean (fallback to highest silhouette)
    recommended = None
    if stabs:
        recommended = sorted(stabs, key=lambda r: r["stability_ari_mean"], reverse=True)[0]["k"]
    else:
        recommended = top3[0]["k"] if top3 else None

    _save_json(out / "k_selection.json", {
        "candidates": cand,
        "recommended_k": recommended,
        "rule": "top-3 silhouette, then highest stability_ari_mean",
    })
    typer.echo(f"Saved selection to: {out / 'k_selection.json'} (recommended_k={recommended})")

@app.command()
def train(
    config: Optional[str] = typer.Option(None, "--config"),
    out_dir: str = typer.Option("runs/train", "--out-dir"),
    k: Optional[int] = typer.Option(None, "--k"),
    seed: int = typer.Option(123, "--seed"),
    verbose: bool = typer.Option(False, "--verbose"),
):
    import faiss
    typer.echo(f"faiss GPUs visible: {faiss.get_num_gpus()}")
    """Train FAISS KMeans and save centroids.npy."""
    cfg = _load_config(config)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    spec = _array_spec_from_cfg(cfg)
    X = load_array(spec)
    X32 = ensure_float32_contiguous(X)

    do_norm = bool(cfg.get("normalize", {}).get("enabled", True))
    if do_norm:
        l2_normalize_inplace(X32)

    km_cfg = cfg.get("kmeans", {})
    K = int(k if k is not None else km_cfg.get("k"))
    centroids, obj = train_faiss_kmeans(
        X32,
        k=K,
        niter=int(km_cfg.get("niter", 25)),
        nredo=int(km_cfg.get("nredo", 1)),
        seed=seed,
        max_points_per_centroid=int(km_cfg.get("max_points_per_centroid", 5000)),
        use_gpu=bool(km_cfg.get("use_gpu", False)),
        verbose=verbose,
    )
    np.save(out / "centroids.npy", centroids)
    _save_json(out / "train_meta.json", {
        "K": K, "obj_last": obj, "normalized_input": do_norm,
        "centroids": "centroids.npy"
    })
    typer.echo(f"Saved: {out / 'centroids.npy'}")

@app.command()
def export(
    config: Optional[str] = typer.Option(None, "--config"),
    out_dir: str = typer.Option("runs/export", "--out-dir"),
    centroids: str = typer.Option(..., "--centroids"),
    topm: int = typer.Option(0, "--topm"),
    full: bool = typer.Option(False, "--full"),
    dtype: str = typer.Option("float32", "--dtype"),   # per your preference
    batch: int = typer.Option(20000, "--batch"),       # per your preference
    backend: ExportBackend = typer.Option("cpu", "--backend"),
    device: str = typer.Option("cuda:0", "--device"),
):
    """Export cosine distances: top-m and/or full NxK memmap."""
    cfg = _load_config(config)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    spec = _array_spec_from_cfg(cfg)
    X = load_array(spec)
    X32 = ensure_float32_contiguous(X)

    C = np.load(centroids).astype(np.float32, copy=False)

    # normalize both for cosine
    l2_normalize_inplace(X32)
    l2_normalize_inplace(C)

    if topm > 0:
        export_topm_cpu(X32, C, m=topm, out_dir=out, batch=batch)
        typer.echo(f"Saved top-m to: {out}")

    if full:
        out_path = out / f"cosdist_{X32.shape[0]}x{C.shape[0]}.{dtype}.mmap"
        if backend == "cpu":
            export_full_memmap_cpu(X32, C, out_path=out_path, batch=batch, out_dtype=dtype)
        else:
            export_full_memmap_gpu(X32, C, out_path=out_path, batch=batch, out_dtype=dtype, device=device)
        typer.echo(f"Saved full memmap to: {out_path}")

@app.command()
def run(
    config: str = typer.Option(..., "--config"),
    out_dir: str = typer.Option("runs/run1", "--out-dir"),
    seed: int = typer.Option(123, "--seed"),
):
    """One-shot: check -> select-k -> train -> export (based on config)."""
    cfg = _load_config(config)
    base = Path(out_dir); base.mkdir(parents=True, exist_ok=True)

    # 1) check
    typer.echo("==> check")
    check(config=config, out_dir=str(base / "check"), sample_size=int(cfg.get("normalize", {}).get("sample_size", 20000)), seed=seed)

    # 2) select-k (optional)
    recommended_k = None
    if cfg.get("k_select", {}).get("enabled", True):
        typer.echo("==> select-k")
        select_k_cmd(config=config, out_dir=str(base / "select_k"), seed=seed, verbose=False)
        sel = json.loads((base / "select_k" / "k_selection.json").read_text(encoding="utf-8"))
        recommended_k = sel.get("recommended_k")

    # 3) train
    typer.echo("==> train")
    k_train = recommended_k if recommended_k is not None else int(cfg.get("kmeans", {}).get("k"))
    train(config=config, out_dir=str(base / "train"), k=k_train, seed=seed, verbose=False)

    # 4) export
    typer.echo("==> export")
    export_cfg = cfg.get("export", {})
    topm = int(export_cfg.get("topm", 0))
    full_cfg = export_cfg.get("full", {})
    full = bool(full_cfg.get("enabled", False))
    dtype = str(full_cfg.get("dtype", "float32"))
    batch = int(full_cfg.get("batch", 20000))
    backend = str(full_cfg.get("backend", "cpu"))
    device = str(full_cfg.get("device", "cuda:0"))

    export(
        config=config,
        out_dir=str(base / "export"),
        centroids=str(base / "train" / "centroids.npy"),
        topm=topm,
        full=full,
        dtype=dtype,
        batch=batch,
        backend=backend,  # cpu/gpu
        device=device,
    )
    typer.echo(f"Done. Outputs in: {base}")