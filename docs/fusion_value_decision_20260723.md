# Fusion value decision (arguments-only protocol)

## Decision

Retain the validation-selected lightweight fusion setting for the locked final
rerun:

- `n_fusion = 1`
- `heads = 8`
- `fusion_dropout = 0.2`
- `lr_fusion = 5e-5`

The evidence supports describing fusion as a modest, metric-dependent
improvement rather than a primary contribution. The architecture decision uses
validation data only. Held-out test results are contextual and must not be used
to reverse that decision.

## Validation-only comparison

All values are mean ± population standard deviation over seeds 0, 1, and 2.
The no-fusion validation metrics were reconstructed from the saved validation
prediction bundles produced by the same arguments-only dataset, splits,
backbone, training schedule, checkpoint rule, and seeds.

| Variant | Accuracy | F1_pos | Macro-F1 | wF1 | AP | AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Locked fusion | 0.7602 ± 0.0110 | 0.6516 ± 0.0133 | 0.7339 ± 0.0014 | 0.7275 ± 0.0033 | 0.6705 ± 0.0175 | 0.8393 ± 0.0033 | 0.1834 ± 0.0037 |
| No fusion | 0.7449 ± 0.0062 | 0.6780 ± 0.0098 | 0.7333 ± 0.0066 | 0.7134 ± 0.0050 | 0.6460 ± 0.0096 | 0.8331 ± 0.0018 | 0.1690 ± 0.0112 |
| Fusion − no fusion | +0.0153 | −0.0264 | +0.0006 | +0.0141 | +0.0246 | +0.0062 | +0.0143 |

Fusion improves the frozen primary selection metric, AP, by 2.46 percentage
points. It also improves accuracy, weighted F1, and AUC, but reduces positive
class F1 and worsens calibration. With only three seeds, these differences
should not be described as statistically significant.

## Historical test context

The earlier arguments-only test comparison used the legacy two-block fusion,
not the locked one-block fusion. It is retained only as context:

- accuracy: +1.25 percentage points with legacy fusion;
- F1_pos: +0.32 points;
- Macro-F1: +0.86 points;
- AP: −0.07 points;
- AUC: −0.12 points;
- ECE: +2.78 points (worse).

This confirms that the benefit is not uniformly large across metrics. The final
paper should report the no-fusion ablation transparently and avoid framing
fusion as the dominant source of performance.

Machine-readable evidence is stored in
`results/audit/fusion_value_analysis.json`.
