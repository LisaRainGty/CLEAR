# Paper experiments checklist (sources_only)

All rows use the same supervised jsonl and three-source evidence.

## RQ1 — Main comparison (Table 3)

- CLAIMARC forward classifier: `c6_canon` × seeds 0/1/2
- (A) ESIM, Decomposable Attention (dam), BERT-NLI
- (B) TextCNN, BiLSTM, BERT-CLS, RoBERTa-CLS
- (C) BGE-frozen LR / SVM / MLP / kNN
- (D) LLM zero/few-shot + Qwen2.5-7B LoRA SFT

Scripts: `run_campaign6`, `scripts/run_baselines.sh`, `run_llm_baselines`

## RQ3 — Cross-domain (Tables 4–5)

- Leave-one-category × 10 (CLAIMARC + BERT/RoBERTa/ESIM)
- Leave-20-streamers × 3 seeds
- Library injection curve (`xdom_inject`)

## RQ2 — Geometry (Table 6)

- Variants: none / supcon / racl × 3 seeds (`run_geom_campaign`)

## RQ4 — Ablations (Tables 7–10)

- Core: no_cl, no_fusion, no_weight, attrblock, lora, bert, concat head (`c6_*`, `c8_concat`)
- Streams: claim_only, evid_only, joint (`c7_*`, `c8_joint`)
- RACL mining: hardpos, Kp, Kn, negfilter, no_classbal
- Reliability counterfactuals / c-formula: `c6_cf_*`
- Selective prediction / calibration figures

## Canonical hyperparameters

See `configs/paper_canonical.yaml`.
