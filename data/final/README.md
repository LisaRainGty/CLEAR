# data/final — 数据集与变体

数据集及其重建/修复链路产物(本地保留,**不入 git**,约 8.5GB)。从
`../claimarc/data/final` 迁入(同盘 `mv`):

```bash
mv "../claimarc/data/final/"* "data/final/"
```

- `dataset.jsonl` — `final.join_split` 产出的基础数据集。
- `repaired_v1/` — 全对重建 + LLM/VLM 评审链路的全部产物,包括:
  - `stateful_proposal_dataset_v2_FULLPOOL_*`(`build_stateful_proposal_dataset_v2`)
  - `dataset_planbaseline_duallabel_FULLPOOL_*`(`build_plan_label_weights_v1`)
  - `dataset_objective_negatives_v1_20260615.jsonl`(`build_objective_negative_dataset_v1`)

最终训练集 `dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl` 由上述
plan-baseline FULLPOOL supervised 与 objective negatives **拼接**而成,并置于 `data/` 根
(模型代码默认从 `data/dataset_*.jsonl` 读取)。完整链路见 `docs/DATA_PIPELINE.md`。
