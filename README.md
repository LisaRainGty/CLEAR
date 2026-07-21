# CLEAR — Consumer-perceived Livestream Misleading Claim Recognition

Retrieval-augmented contrastive learning for **attribute-level consumer-perceived
misleading claims** in livestream e-commerce (CLAIMARC / CLEAR).

> **Fair protocol (this release):** every trainable system uses the **same**
> supervised split and **three-source product evidence only**
> (`evidence_params` + `evidence_ocr` + `evidence_vlm`). LLM `arguments` are
> **not** used in any paper table.

## Cite

Paper draft and experiment chapter: `docs/实验结果与分析.md` (and `.docx`).
English LaTeX: `paper/claimarc_aaai.tex`.

## Layout

```
main/
├── README.md
├── requirements.txt / requirements_lock.txt / env.example.sh
├── configs/paper_canonical.yaml
├── scripts/                 # one-click paper reproduction
│   ├── run_paper_all.sh
│   └── run_baselines.sh
├── src/                     # pipeline + models
├── data/
│   ├── dataset_*_supervised_*.jsonl   # training set (required)
│   ├── dataset_*_all_*.jsonl          # retrieval pool
│   └── raw/ -> symlink to local raw corpus (~34GB; optional for train-from-jsonl)
├── results/                 # new sources_only runs land here
│   └── legacy_pre_sources_only/       # previous runs (reference only)
├── embeddings/              # RQ2 bundles (regenerated)
├── paper/                   # tex/md/figs
└── docs/                    # pipeline + experiment extract
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example.sh env.sh && source env.sh
# optional offline encoders:
# export CLAIMARC_BGE_PATH=/path/to/bge-large-zh-v1.5
# export CLAIMARC_BERT_PATH=/path/to/bert-base-chinese
# export CLAIMARC_ROBERTA_PATH=/path/to/chinese-roberta-wwm-ext
```

Hardware: single 24GB GPU (RTX 4090 class) as in the paper.

## Data

See [`data/README.md`](data/README.md). Training needs only:

- `data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl` (4,883 pairs)
- `data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_all_20260615.jsonl` (retrieval pool)

Splits are embedded (`split` = train/val/test; grouped by `room_id`, 70/10/20).
Evidence tokenization uses **sources_only** (no `arguments` field required).

## Replicating (paper tables)

```bash
source env.sh
bash scripts/run_paper_all.sh
```

Or step-by-step:

| Step | Command | Paper tables |
|------|---------|--------------|
| Canonical + xdom + core ablations | `python -m models.run_campaign6` | Table 3 headline, 4–5, 7, 10, 12 |
| Stream / RACL mining ablations | `python -m models.run_campaign7` | Table 8–9 |
| Concat / joint | `python -m models.run_campaign8` | Table 7–8 |
| Baselines A–C | `bash scripts/run_baselines.sh` | Table 3 |
| Geometry RQ2 | `python -m models.run_geom_campaign ...` | Table 6 |
| LLM baselines | `python -m models.run_llm_baselines` | Table 3 (D); needs API key |

All campaign scripts force `--evidence_policy sources_only`.

## Results mapping

| File | Content |
|------|---------|
| `results/campaign6_results.jsonl` | CLAIMARC canon, xdom folds, core ablations |
| `results/campaign7_results.jsonl` | stream + RACL mining |
| `results/campaign8_results.jsonl` | concat / joint |
| `results/baselines_results.jsonl` | Table 3 baselines |
| `results/artifacts/geom2.json` | RQ2 geometry |
| `results/legacy_pre_sources_only/` | **pre-fair** numbers (do not mix with new runs) |

## License / privacy

Training jsonl contains livestream/product text; respect platform terms if redistributing.
Raw multimodal corpus is local-only (symlink); not required to retrain from the released jsonl.

## Security note

Never commit `env.sh`, API keys, or SSH passwords.
