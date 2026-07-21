# Remote fair-rerun record

The fresh paper rerun uses one NVIDIA A30 (24 GB), driver 470.57.02, Python
3.9.16 and PyTorch 2.0.0+cu117.  The isolated working directory is
`/root/CLEAR_run`; no password, token or API key is stored in this repository.

The frozen contract is:

- dataset SHA-256 `1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76`;
- existing train/val/test and room-disjoint splits only;
- `sources_only`: PARAM + OCR + VLM concatenation for every model;
- zero argument fields consumed;
- seeds 0, 1 and 2 for main comparisons and model ablations;
- validation-only checkpoint and threshold selection;
- full-FT CLAIMARC uses micro-batch 12 and accumulation 3 (effective batch 36).

Before the queue was launched, `scripts/smoke_gpu.py` completed a forward,
backward and optimizer step on the 12 longest evidence rows.  Evidence was
truncated only at the frozen 384-token model limit; peak allocated CUDA memory
was 13,553.9 MiB.  The full suite is resumable per job:

```bash
cd /root/CLEAR_run
CLAIMARC_PYTHON=/root/claimarc_venv/bin/python \
  /root/claimarc_venv/bin/python scripts/run_paper_suite.py \
  --stages all --execute --continue-on-error --quiet
```

The API-based commercial/hosted LLM stage is separate because it can incur
external cost and requires provider credentials:

```bash
/root/claimarc_venv/bin/python scripts/run_paper_suite.py \
  --stages llm --execute --continue-on-error
```

The full in-domain and cross-domain matrix contains 37,190 row-level requests
before caching.  The request cache hashes the model, complete messages,
temperature and token limit; therefore only byte-identical requests are reused.
An offline prompt audit found 26,968 unique payloads, which is the upper bound
before any pre-existing cache hits.  This optimisation changes neither prompts
nor fold-specific few-shot examples and cannot transfer labels or thresholds
between folds.

Hosted-LLM jobs are successful only when every validation and test response is
parsed (`n_err_val = n_err_test = 0`).  A transient failure makes the job
non-zero and resumable; the next invocation reuses exact successful payloads and
requests only the missing cache entries.  Missing responses are never replaced
by a default score in a paper table.

The output caps are fixed per endpoint at 320 tokens for Qwen-Flash/GPT-5.4,
1,024 for Gemini-3.5-Flash and 4,096 for Kimi-K2.6.  Longest-prompt probes showed
that the two reasoning endpoints could otherwise exhaust the cap before closing
the required JSON object.  The prompt, evidence, temperature and output schema
remain identical; caps were chosen solely for parse completeness before looking
at evaluation metrics.

Each completed job writes an independent log, result JSONL, status JSON and
prediction/embedding bundle under `results/fair_rerun/` and
`embeddings/fair_rerun/`.  `scripts/aggregate_paper_results.py` compiles these
artifacts into `results/fair_rerun/paper_tables.md` without filling missing rows
from historical experiments.
