# RACL validation-only refinement v2

The first no-fusion diagnostic grid completed all 18 runs without failures or
test/RKC access. Its selected RACL configuration increased mean validation AP
from `0.6417` (no RACL) to `0.6497`, but reduced positive-class F1 from
`0.6945` to `0.6698`.

The epoch logs show that seeds 0 and 2 retained a pre-RACL warm-up checkpoint.
Only seed 1 selected a contrastive-stage checkpoint, where AP improved but
positive-class F1 declined. The first RACL update was relatively large compared
with the classification loss, and later epochs strongly overfit.

This refinement therefore changes only predeclared training-schedule and RACL
hyperparameters:

- two warm-up epochs and four RACL epochs instead of three plus six;
- `lambda_cl` in `{0.05, 0.10}`;
- temperature in `{0.10, 0.15}`;
- `Kp` in `{1, 3}` with `Kn=5`;
- self-positive exclusion remains enabled for every selectable RACL candidate.

A no-RACL run with the same shortened schedule is diagnostic only. Every
candidate uses the same frozen arguments-only dataset, splits, optimizer,
effective batch size, no-fusion architecture and seeds 0/1/2. Selection uses
mean validation AP, positive-class F1 as tie-breaker and population SD. The
frozen first-phase winner is included in the final cross-phase ranking. Test and
RKC access remain forbidden.
