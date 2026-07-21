# data/processed — 阶段中间产物

各流水线阶段的中间产物(本地保留,**不入 git**,约 6.4GB)。从 `../claimarc/data/processed`
迁入(同盘 `mv`):

```bash
mv "../claimarc/data/processed/"* "data/processed/"
```

| 子目录/文件 | 来源阶段 |
| --- | --- |
| `stageA/`, `stageA_repaired_v1/` | Stage A 评论属性抽取/标准化 |
| `stageB/`, `stageB_product_v2*/`, `stageB_fullschema_gap/` | Stage B 话术抽取/对齐 |
| `stageC/`, `stageC_neg/`, `stageC_gap/` | Stage C 商品证据/负例证据 |
| `labels.jsonl` | §2 标签引擎输出(y, c) |

阶段↔模块↔产物映射见 `docs/DATA_PIPELINE.md`。
