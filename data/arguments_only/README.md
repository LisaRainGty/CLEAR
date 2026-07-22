# Arguments-only paper dataset

This directory contains the frozen evidence view used by the arguments-only
fair rerun. It is derived from the unchanged 4,883-row paper dataset. Labels,
splits, reliability weights, pair order, and every original field remain byte-
equivalent at the JSON-value level; only `arguments` and
`_argument_generation` are added.

## Leakage-safe generation contract

The generator receives only five label-blind inputs for each pair:

1. attribute name;
2. livestream claim text;
3. PARAM source texts;
4. OCR source texts;
5. VLM source texts.

It never receives labels, split names, reliability weights, sample roles,
consumer comments, audit decisions, or historical arguments. Rows with no raw
source use one fixed evidence-gap statement; all source-bearing rows use the
same deterministic open-source instruction model. Historical partial arguments
are not reused because doing so would mix generators and coverage regimes.

The final release consists of:

- `dataset_arguments_only_20260722.jsonl`: exact training/evaluation dataset;
- `raw_generations.jsonl`: append-only prompts, raw responses, rejected attempts,
  and accepted normalized arguments;
- `MANIFEST.json`: source/output hashes, prompt hash, model snapshot inventory,
  decoding settings, counts, and field coverage;
- `LINEAGE_AUDIT.json`: independent source-to-cache-to-final consistency audit.

Generate and audit from the repository root:

```bash
python scripts/generate_arguments_dataset.py \
  --model-path /path/to/Qwen2.5-32B-Instruct \
  --hash-model-shards

python scripts/audit_arguments_dataset.py \
  --source data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl \
  --dataset data/arguments_only/dataset_arguments_only_20260722.jsonl \
  --cache data/arguments_only/raw_generations.jsonl \
  --manifest data/arguments_only/MANIFEST.json \
  --out data/arguments_only/LINEAGE_AUDIT.json
```

Training never consumes the retained raw sources from this file. Every paper
job is invoked with the explicit `args_only` policy, and the regression test in
`tests/test_evidence_policy.py` verifies that PARAM/OCR/VLM tokens are absent
from the evidence stream while the three argument fields are present.
