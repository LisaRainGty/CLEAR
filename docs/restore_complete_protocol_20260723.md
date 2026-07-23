# Restored CLAIMARC complete protocol

## Decision

The paper main model is restored to the complete configuration that produced
the three-seed test result:

| Model | Seeds | Macro-F1 | Positive F1 | AUPRC | AUROC | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLAIMARC (complete) | 3 | 80.63 ± 0.42 | 74.56 | 75.24 | 90.53 | 82.53 | 13.89 |

The authoritative configuration is `configs/paper_fair.json`. The three
underlying result records are
`results/arguments_only_rerun/jobs/claimarc_canonical_s{0,1,2}.jsonl`.

## Exact model path

- frozen arguments-only dataset SHA-256:
  `b6bc9a91f87da3a489af216a036e4d11bd02d7eb8895e9d7f2cd9d78e26bd618`;
- two fusion blocks, eight attention heads and fusion dropout 0.2;
- full BGE encoder training with encoder LR `1e-5` and head/fusion LR `1e-4`;
- three warm-up epochs followed by six RACL epochs;
- global, class-balanced RACL retrieval;
- `lambda_cl=0.5`, `tau=0.07`, `Kp=3`, `Kn=5`;
- validation-only threshold and checkpoint selection;
- one final evaluation on the 855-row held-out test split;
- seeds 0, 1 and 2.

This command path was deployed immediately after commit `140a564`. The current
runner remains backward compatible: with `configs/paper_fair.json`, none of the
later opt-in flags (`--no_fusion`, `--cl_exclude_self`, `--validation_only`, or
a separate fusion learning rate) is active.

## Why the later result appeared to collapse

The 80.63 headline is test Macro-F1. The later roughly 73.6 figure is validation
Macro-F1 from a different no-fusion tuning protocol. They are not comparable.
The original complete model itself selected checkpoints with validation
Macro-F1 values 0.7295, 0.7440 and 0.7209 (mean 0.7315), while its final test
Macro-F1 values were 0.8100, 0.8072 and 0.8017 (mean 0.8063). The newer
attribute-RACL validation result therefore did not demonstrate a 7-point model
collapse.

There was also a real architecture change: the diagnostic protocol removed
fusion, shortened training from 3+6 to 2+4 epochs, changed `lambda_cl` from 0.5
to 0.1 and `tau` from 0.07 to 0.1, excluded the anchor from retrieval, and
enabled attribute-conditioned retrieval. Those runs remain archived, but they
are no longer the paper main model.

## Reporting rule

Final tables must never mix validation and test metrics. Model selection and
diagnosis use validation only; the locked final model is reported once on test.
All three-seed paper metrics use mean ± sample standard deviation (`ddof=1`).
Validation-only selector manifests retain their originally preregistered
population-SD convention and are not paper result tables.
