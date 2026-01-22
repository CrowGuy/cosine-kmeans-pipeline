# cosine-kmeans-pipeline

Pipeline:
1) check/normalize embeddings
2) FAISS KMeans training
3) save centroids
4) export cosine distances (top-m or full NxK memmap)

## Install (CPU)
```bash
pip install -r requirements.txt
pip install -e .
```

## Install (GPU distance export optional)
```bash
pip install -r requirements-gpu.txt
```

## Run
```bash
kmcp run --config configs/default.yaml --out-dir runs/exp1
```

## Command
- `kmcp check --config ...`
- `kmcp select-k --config ...`
- `kmcp train --config ...`
- `kmcp export --config ... --centroids ... --full --backend cpu|gpu`

---
## 怎麼用（最符合你目前流程）
一鍵跑：
```bash
kmcp run --config configs/default.yaml --out-dir runs/exp1
```

只跑 train + full export（你已經確定 K=1400）：
```bash
kmcp train --config configs/default.yaml --out-dir runs/exp1/train --k 1400
kmcp export --config configs/default.yaml --out-dir runs/exp1/export --centroids runs/exp1/train/centroids.npy --full --dtype float32 --batch 20000 --backend cpu
```

切成 GPU export（要先裝 torch + CUDA）：
```bash
kmcp export --config configs/default.yaml --out-dir runs/exp1/export --centroids runs/exp1/train/centroids.npy --full --dtype float32 --batch 20000 --backend gpu --device cuda:0
```