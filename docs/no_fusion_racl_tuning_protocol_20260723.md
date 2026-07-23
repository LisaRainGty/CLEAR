# No-fusion main model and RACL tuning protocol

## Decision revision

The main CLAIMARC architecture is revised from the previously selected
`fusion_v2` variant to `no_fusion + RACL` at the author's request. RACL is a
mandatory core contribution, not an optional component. The completed
fusion-v2 artifacts remain immutable provenance and are not deleted or
reinterpreted. This revision is prospective for all final reruns.

`no_fusion` still preserves claim–argument interaction through the classifier's
four-tuple representation `[h_claim, h_argument, h_claim-h_argument,
h_claim*h_argument]`; it removes only the learned cross-attention fusion block.

## Why RACL is retuned

Under the frozen arguments-only dataset, the default RACL setting
(`lambda_cl=0.5`, `tau=0.07`, `Kp=3`, `Kn=5`) often selected a checkpoint at
the last warm-up epoch or the first contrastive epoch. Later contrastive epochs
frequently reduced validation AP. This is consistent with an overly strong or
overly sharp contrastive objective after the generated arguments have already
compressed the evidence relation.

Code inspection identified a concrete dilution mechanism in the legacy RACL:
the epoch memory bank contains the anchor itself, and the same record usually
occupies one of the easiest positive-neighbour slots. The revised RACL exposes
an auditable `cl_exclude_self` option that masks an identical `pair_id` before
positive retrieval. The legacy behavior remains a declared control.

## Frozen diagnostic comparison

- Dataset: `data/arguments_only/dataset_arguments_only_20260722.jsonl`
- SHA-256: `b6bc9a91f87da3a489af216a036e4d11bd02d7eb8895e9d7f2cd9d78e26bd618`
- Architecture: `no_fusion` for every candidate
- Seeds: 0, 1, 2
- Split, checkpoint rule, threshold rule, batch budget, encoder and optimizer:
  identical across candidates
- Selection data: validation only; test and RKC access are forbidden
- Standard deviation: population SD (`ddof=0`)

The six predeclared candidates are:

1. No RACL, used only as a diagnostic ablation.
2. Legacy RACL including the anchor itself: `lambda_cl=0.5`, `tau=0.07`,
   `Kp=3`, `Kn=5`.
3. Self-excluded RACL: `lambda_cl=0.2`, `tau=0.10`, `Kp=3`, `Kn=5`.
4. Weak self-excluded RACL: `lambda_cl=0.1`, `tau=0.10`, `Kp=3`, `Kn=5`.
5. Weak self-excluded RACL with a broader neighbourhood: `Kp=5`, `Kn=10`.
6. Weak self-excluded RACL with `c>=0.2` for anchors, positives and negatives.

## Selection rule

No-RACL is never selectable as the main model. Among RACL-enabled candidates,
validation AP is primary and positive-class F1 is the tie-breaker. The report
still records matched-seed AP wins and all metric deltas against no-RACL so the
module's empirical contribution remains transparent. If this diagnostic grid
does not produce a satisfactory RACL improvement, a second validation-only
refinement must be preregistered around the best RACL region; test access
remains forbidden until the final RACL configuration is locked.
