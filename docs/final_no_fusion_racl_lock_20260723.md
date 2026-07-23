# Final no-fusion + RACL architecture lock

The final main architecture is fixed before any new test-set evaluation:

- evidence policy: `args_only`;
- architecture: `no_fusion`;
- RACL retrieval: attribute-conditioned, class-balanced and self-excluded;
- schedule: 2 warm-up epochs and 4 contrastive epochs;
- RACL weight/temperature: `lambda_cl=0.10`, `tau=0.10`;
- neighbours: `Kp=3`, `Kn=5`;
- seeds: 0, 1 and 2.

The validation-only selector chose
`racl_attr_l010_t010_k3_5_w2c4` with AP `0.6671±0.0097`.
The matched no-RACL diagnostic reached AP `0.6625±0.0051`; the matched-seed
mean AP delta is `+0.0047`, with RACL winning two of three seeds. Positive-F1
was `0.6732±0.0199` versus `0.6754±0.0201` (`-0.0023`).

The selection manifest is
`results/arguments_only_no_fusion_racl_attr_refine_v3/racl_no_fusion_selection.json`
with SHA-256
`bc528d11c43fe8d49bf1e69302b28698a3b7695216930d2201eb302232a7e99b`.
It records `test_metrics_accessed=false`. No further RACL tuning is allowed
without a new preregistered validation-only protocol.

All subsequent CLAIMARC-dependent experiments, including cross-domain folds,
must use this lock. Table 8 retains one frozen with-fusion diagnostic, while
the canonical model remains no-fusion. Table 9 uses global RACL retrieval as
the mining ablation because attribute-conditioned retrieval is now canonical.
