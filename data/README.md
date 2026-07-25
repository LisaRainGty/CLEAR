# 数据目录

本目录保存 CLAIMARC 从原始观测到论文训练集的完整数据链。请先阅读
`../docs/DATA_PROVENANCE.md`。

- `archives/raw.tar.zst`：原始评论、字幕和商品图片的完整归档。
- `index/`：商品索引。
- `archives/processed.tar.zst`：正式 Stage A/B/C、product-v2、labels 与 gap/negative 中间产物的完整归档。
- `final/`：流水线 join 结果、冻结重建队列/44 个评审输出与 FULLPOOL 处理阶段。
- `dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl`：论文模型唯一默认训练文件，4,883 行。

监督集 SHA-256：

```text
1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76
```

最终数据中非空 `arguments` 记录为 0，且 4,883 条记录均不存在
`arguments` 字段。发布树不保留 arguments 数据变体；raw、processed 归档内也不含
名称为 `args` 或 `arguments` 的历史数据变体。
不要在冻结文件上原地修订；任何标签、三源证据或 split 变化都必须使用新版本名、更新配置和哈希。

无需解包的完整性校验：

```bash
python scripts/unpack_data_archives.py --verify-only
```

如需从 raw 开始运行 Stage A/B/C，执行 `python scripts/unpack_data_archives.py`；
解包器会拒绝覆盖已存在的数据目录。
