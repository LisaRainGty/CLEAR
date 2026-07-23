# Final RACL attribute-conditioned refinement v3

Refinement v2 found a balanced global-retrieval setting
(`lambda_cl=0.10`, `tau=0.10`, `Kp=3`, `Kn=5`) that slightly improved both
validation AP and positive-class F1 over its matched no-RACL control while
reducing ECE from `0.1511` to `0.1327`. The AP-primary selector instead chose
`Kp=1`, whose AP gain was larger but whose F1 gain was not.

The remaining diagnosis is that global RACL can retrieve positives and
hard negatives from unrelated product attributes. Since each generated
argument is explicitly conditioned on a claim attribute, this may inject
semantic noise. This final, validation-only refinement compares:

- the matched shortened-schedule no-RACL diagnostic;
- the balanced global-retrieval anchor;
- attribute-conditioned retrieval at the balanced setting;
- attribute-conditioned retrieval with softer temperature/lower weight;
- attribute-conditioned retrieval with one positive neighbour.

All candidates use the immutable arguments-only dataset, no-fusion
architecture, self-positive exclusion, identical splits, seeds, optimizer,
effective batch, checkpoint and threshold rules. No-RACL is never selectable.
The frozen refinement-v2 winner is included in the cross-phase AP-primary
ranking. Test and RKC access remain forbidden.
