# CLAIMARC Data Reconstruction Notes (2026-06-12)

## Why the original labels were noisy

- The documented Stage B rule is conceptually valid for consumer-perception risk:
  a pair is positive when at least one aligned review is negative.
- The actual final dataset expands this rule into a very broad negative class:
  all pairs without aligned negative reviews become `y=0`.
- Funnel audit shows the risk:
  - final records: 16,679
  - claimful records: 3,514
  - original positives: 643
  - claimful original negatives: 2,871
  - no-claim pairs: 13,165
- Therefore many `y=0` records mean "no observed aligned complaint", not
  "claim is supported by product evidence".

## Reconstruction protocol

1. Keep only claimful product-attribute pairs as the broad candidate pool.
2. Run claim-evidence adjudication that sees only:
   - category and attribute,
   - grounded livestream claim text,
   - product evidence from params/OCR/VLM.
3. Combine the original consumer-perception weak label with the blind
   claim-evidence state:
   - confirmed/consumer-risk positives are kept as positives,
   - evidence-risk weak negatives are relabeled as silver positives,
   - supported clean cases are kept as negatives,
   - bad claims and ambiguous negatives are dropped.
4. Repair source0 pairs before adjudication:
   - if params/OCR/VLM evidence was empty, attach low-confidence raw
     product-index context from the original product title and parameter table;
   - no reviews, labels, rationales, or external search are included.

## Candidate datasets

| dataset | n | labels | source0 | lightweight OOF |
|---|---:|---:|---:|---|
| `dataset_hq_adjudicated_v1` | 1,424 | 900/524 pos/neg | mixed | AP 0.7364, AUROC 0.5890, Macro-F1 0.5613 |
| `dataset_hq_broad_adjudicated_v1` | 3,100 | 2,125/975 pos/neg | 1,241 | AP 0.8764, AUROC 0.7530, Macro-F1 0.6637 |
| `dataset_hq_broad_enriched_adjudicated_v1` | 3,083 | 1,538/1,545 pos/neg | 0 | AP 0.6171, AUROC 0.6092, Macro-F1 0.5662 |
| `dataset_hq_broad_enriched_adjudicated_strict_v1` | 2,182 | 1,325/857 pos/neg | 0 | AP 0.7636, AUROC 0.7011, Macro-F1 0.6385 |
| `strict_plus_perception` | 2,395 | 1,538/857 pos/neg | 0 | AP 0.7713, AUROC 0.6838, Macro-F1 0.6184 |

## Current interpretation

- The non-enriched broad full set is highly learnable, but all source0 selected
  samples are positive, so it risks a missing-evidence shortcut.
- The enriched full set removes that shortcut and reaches the requested 3k+
  scale, but becomes noisier after low-risk negatives and weak perception
  positives are included.
- The enriched strict set is the cleanest high-confidence core and should be
  the primary candidate for rigorous model comparison.
- The 3k+ enriched full set remains useful as a robustness/noisy-silver setting.

## Paper framing

The data contribution should be framed as dual-channel silver supervision:
consumer-perception distant labels are preserved, but blind claim-evidence
adjudication separates clean negatives, evidence-risk positives, perception-only
positives, and ambiguous drops. This is more defensible than treating the
original `otherwise y=0` rule as a factual no-risk label.
