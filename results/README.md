# Results ↔ paper tables (sources_only rerun)

After `bash scripts/run_paper_all.sh`:

| Paper | Metric source |
|-------|----------------|
| Table 1–2 (stats) | Derived from supervised jsonl |
| Table 3 main | `campaign6` tag `c6_canon` + `baselines_results.jsonl` + LLM |
| Table 4–5 xdom / inject | `results/xdom_c6/` + `xdom_agg` / `xdom_inject` |
| Table 6 geometry | `artifacts/geom2.json` |
| Table 7–10 ablations | `c6_*`, `c7_*`, `c8_*` tags |
| Table 11–12 sweeps | `c6_cf_*` + LoRA sweep scripts |

`legacy_pre_sources_only/` holds older numbers (mixed evidence / args_only headline).
**Do not cite legacy for the fair sources_only release.**
