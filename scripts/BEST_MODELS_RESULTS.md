# Best Performing Models: RAVEN & I-RAVEN
Collected via `./scripts/get_best_metrics.sh` (validation accuracy at best epoch).
## RAVEN (trained until 404 epochs)
| Model | Val Acc | Best Epoch | Config |
|-------|---------|------------|--------|
| **object_vsa_gnn_frontend_gnn_object_only_unified** | **96.45%** | - | object_only, unified_graph |
| object_vsa_gnn_frontend_gnn_granular_object_only_unified | 96.43% | - | granular, object_only, unified_graph |
| object_vsa_gnn_frontend_gnn_object_only | 81.09% | - | object_only |
| object_vsa_gnn_frontend_gnn_granular_object_only | 80.31% | - | granular, object_only |
| object_vsa_gnn_frontend_gnn_unified | 48.66% | - | unified_graph |
| object_vsa_gnn_frontend_gnn_granular_unified | 28.09% | - | granular, unified_graph |
| object_vsa_gnn_frontend_gnn | 25.57% | - | baseline |
| object_vsa_gnn_frontend_gnn_granular | 26.54% | - | granular |
**Best RAVEN model:** `experiments/object_vsa_gnn_frontend_gnn_object_only_unified/lr0.001_embedding_emb256_h384_ep404/model_best.pt`
---
## I-RAVEN (trained until 406 epochs)
| Model | Val Acc | Best Epoch | Config |
|-------|---------|------------|--------|
| **object_vsa_gnn_frontend_gnn_granular_object_only** | **68.83%** | - | granular, object_only |
| object_vsa_gnn_frontend_gnn_object_only | 68.44% | - | object_only |
| object_vsa_gnn_frontend_gnn_granular_object_only_unified | 58.54% | - | granular, object_only, unified_graph |
| object_vsa_gnn_frontend_gnn_object_only_unified | 48.28% | - | object_only, unified_graph |
| object_vsa_gnn_frontend_gnn_unified | 35.39% | - | unified_graph |
| object_vsa_gnn_frontend_gnn_granular_unified | 36.17% | - | granular, unified_graph |
| object_vsa_gnn_frontend_gnn | 34.38% | - | baseline |
| object_vsa_gnn_frontend_gnn_granular | 30.84% | - | granular |
**Best I-RAVEN model:** `experiments/object_vsa_gnn_frontend_gnn_granular_object_only/lr0.001_embedding_emb256_h384_ep406/model_best.pt`
---
## Test Evaluation Scripts
### RAVEN
```bash
./scripts/test_raven.sh
```
Uses best RAVEN checkpoint and `data/RAVEN-10000`. Reports per-constellation test accuracy.
Override paths:
```bash
CHECKPOINT=/path/to/model_best.pt DATASET_ROOT=/path/to/RAVEN-10000 ./scripts/test_raven.sh
```
### I-RAVEN
```bash
./scripts/test_i_raven.sh
```
Uses best I-RAVEN checkpoint and `data/I-RAVEN-10000`. Reports per-constellation test accuracy.
Override paths:
```bash
CHECKPOINT=/path/to/model_best.pt DATASET_ROOT=/path/to/I-RAVEN-10000 ./scripts/test_i_raven.sh
```
### Direct Python call
```bash
python3 eval_gnn_test.py --checkpoint <path/to/model_best.pt> --dataset-root data/RAVEN-10000
python3 eval_gnn_test.py --checkpoint <path/to/model_best.pt> --dataset-root data/I-RAVEN-10000
```
---
## Constellations (identical for both datasets)
1. center_single
2. distribute_four
3. distribute_nine
4. left_center_single_right_center_single
5. up_center_single_down_center_single
6. in_center_single_out_center_single
7. in_distribute_four_out_center_single