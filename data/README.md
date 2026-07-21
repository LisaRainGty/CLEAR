# Data

## Released for model reproduction

| File | Rows | Role |
|------|------|------|
| `dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl` | 4,883 | Paper train/val/test |
| `dataset_duallabel_FULLPOOL_PLUS_OBJNEG_all_20260615.jsonl` | 11,513 | Full pool (retrieval / unlabeled roles) |

### Critical fields

- `pair_id`, `room_id`, `category`, `attribute_id` / `attribute_name`
- `split` ∈ {train, val, test} (room-grouped 70/10/20)
- `y` (label), `c` (reliability weight)
- `claim.segments[].text` (streamer pitch)
- `evidence_params` / `evidence_ocr` / `evidence_vlm` (**only** evidence used in paper runs)
- `sample_role`, `contrastive_mask`

### Fair protocol

Paper experiments set `evidence_policy=sources_only`. The optional LLM
`arguments` field is **not** used and may be absent.

## Raw corpus

`raw/` may be a symlink to the local ~34GB corpus (comments, SRT, product images).
End-to-end pipeline reproduction needs Matpool LLM/VLM credentials; see
`docs/DATA_PIPELINE.md`. Reviewers reproducing **Table metrics** only need the
jsonl files above.
