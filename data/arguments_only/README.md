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
same fixed MatPool hosted model ID, endpoint, temperature, and token cap.
Provider determinism is not assumed: the exact prompts, responses, and final
arguments are frozen and hashed. Historical partial arguments are not reused
because doing so would mix generators and coverage regimes.

Accepted supporting/refuting arguments must contain a direct text anchor from
PARAM/OCR/VLM. Any number in those fields must also occur in the raw source,
and speculative phrases are rejected. Every delimiter-separated subclaim is
checked, while numeric contradictions require matching units and measurement
dimensions; unit conversion is prohibited. Each field is prompted to stay within
120 Chinese characters and is rejected above 160 characters, preventing raw
evidence dumps while retaining a small validation tolerance. Failed generations
stay in the raw cache for auditability and are retried; they never enter the
final dataset. Identical non-empty support and refute fields are also rejected:
when one source bears differently on separate subclaims, both relationships must
be stated explicitly instead of duplicating an unqualified quotation. Retry
prompts are failure-specific and carry their attempt number so deterministic
decoding cannot repeat an identical correction prompt.

The final release consists of:

- `dataset_arguments_only_20260722.jsonl`: exact training/evaluation dataset;
- `raw_generations.jsonl`: append-only prompts, raw responses, rejected attempts,
  and accepted normalized arguments. Every attempted request stores the exact
  prompt and its SHA-256, the sanitized provider response, and token usage;
- `MANIFEST.json`: source/output hashes, prompt hash, model snapshot inventory,
  decoding settings, counts, and field coverage;
- `LINEAGE_AUDIT.json`: independent source-to-cache-to-final consistency audit.

Generate and audit from the repository root:

```bash
# Set MATPOOL_API_KEY only in the process environment; never put it in a file.
python scripts/generate_arguments_dataset.py \
  --backend matpool \
  --model-id Qwen3.7-Max \
  --revision matpool_live_20260722 \
  --input-price-per-million 9.6 \
  --output-price-per-million 28.8

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
