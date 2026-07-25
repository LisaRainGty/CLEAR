# 数据来源与加工链

## 固化范围

`main/data` 保留从原始观测到论文训练集的四层数据。表中规模为清理前完整快照;
对外发布清单只保留论文血缘所需的文件,历史训练检查点、旧结果和数据集探索变体不纳入数据层。

| 层 | 内容 | 当前规模/文件数 | 本次快照来源 |
|---|---|---:|---|
| `archives/raw.tar.zst` | 评论、直播字幕、商品图片 | 1,365,590,794 逻辑字节 / 15,952 文件 | 原项目当前 raw |
| `index/` | `product_index.json` | 4,844,266 字节 / 1 文件 | 同盘完整上游项目 `../claimarc` |
| `archives/processed.tar.zst` | 正式 Stage A/B/C、product-v2、修复 Stage-A、gap/negative 与 labels | 892,901,382 逻辑字节 / 2,178 文件 | `../claimarc/data/processed` 经血缘清理 |
| `final/` | 基础 join、冻结队列/评审、FULLPOOL 处理阶段 | 334,703,284 字节 / 101 文件 | `../claimarc/data/final` 经血缘清理 |
| 论文 supervised JSONL | 唯一模型输入数据 | 19,199,663 字节 / 4,883 条 | 固化 final snapshot |

原 `claimarc_final_副本/data/processed`、`index`、`final` 在审计时基本为空，而项目 README 指向 `../claimarc/data/<layer>` 迁移。因此本快照从同一磁盘上的完整上游项目补齐这些层。raw 和 processed 用确定顺序 tar+zstd 归档，避免 Git/LFS 存储上万个小对象；`data/archives/MANIFEST.json` 记录归档哈希、文件数、逻辑字节数与排序内容树哈希。流式校验已通过。

| 归档 | 归档 SHA-256 | 排序内容树 SHA-256 |
|---|---|---|
| raw | `53eecb89b887d4d6c4d5de849b77109e3751443006183f132576ae485ee88774` | `f6614c281f79b4772f1fba382c30cb9c0124b972388ffdef1e4ad73b4e911ab1` |
| processed | `d8e4025c340b23830eb8fc9a8c5d94dd8844c071933d71fd0fdb881f3e347e4b` | `4aa38a7de4f9984f12c32cca3a50e57d631a91138cbe7f8bd889211d6e791390` |

关键哈希：

```text
supervised JSONL
1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76

all JSONL
701929de68f779016544150a8f2e09639c644d23f816608541c9aad33c43d345

pipeline data/final/dataset.jsonl
c768bad80be02fe0fe8bdfe305b00bd7e640e0e32f12778695dd8afa494c35fb
```

论文训练集是下列两个冻结组件的按字节连接：

| 组件 | 行数 | SHA-256 |
|---|---:|---|
| `dataset_planbaseline_duallabel_FULLPOOL_supervised_20260614_stagec.jsonl` | 2,278 | `0cb580f5df791aebfa963c9e84a701c96f7aa4c3d85d337dfb65839f11b60492` |
| `dataset_objective_negatives_v1_20260615.jsonl` | 2,605 | `7df3f9ac18c6691410489322c129600d3ba528190c0f3c9757351a57b6b151c9` |
| 最终 supervised | 4,883 | `1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76` |

临时目录全链重建检查已通过：stateful、plan label、Stage-C 三源合并、
objective negatives 与最终 supervised 共 10 个产物的行数和 SHA-256 全部一致。

## 流水线映射

```text
raw/comment + raw/srt_cut + raw/product_images
        │
        ├─ Stage A: 属性 schema、aspect 聚类与规范化
        ├─ Stage B: 主播 claim 抽取、passage 构造、属性对齐
        └─ Stage C: 参数、OCR、VLM 商品事实抽取
                         │
processed/labels.jsonl ─┤
                         ▼
              final/join_split.py
                         ▼
       data/final/dataset.jsonl + 冻结重建队列/评审输出
                         ▼
stateful FULLPOOL → plan label → Stage-C PARAM/OCR/VLM 合并
                         ▼
     plan supervised + objective negatives
                         ▼
dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl
                         ▼
       固定 [PARAM]→[OCR]→[VLM] 输入的论文模型训练
```

代码对应 `src/stage_a`、`src/stage_b`、`src/stage_c`、`src/labels`、`src/final` 和 `src/data_quality`。
`src/data_quality/rebuild_paper_dataset.py` 从有序的 44 个冻结评审输出开始执行论文唯一有效的
FULLPOOL 后续链路。历史候选数据集、`cleancl*`、`v2/` 训练产物和 `*_args*` 文件
不属于这条血缘,将不出现在最终发布包。

## 冻结边界

- 原始评论、字幕和商品图片是原始观测层。
- Stage A/B/C 产物保留每步处理结果。
- raw/processed 默认以两个无损归档发布；`python scripts/unpack_data_archives.py`
  可恢复原目录，`--verify-only` 可在不解包时验证每个载荷。
- `full_pair_reconstruction_queue_v1_20260614.jsonl` 和其 44 个有序评审 JSONL 是外部模型调用后的冻结边界。
- 从该边界到 4,883 条论文训练集可在本地逐字节重建;检查命令为
  `PYTHONPATH=src python -m data_quality.rebuild_paper_dataset`。
- 最终数据中非空 `arguments` 记录数为 0;训练不读取也不生成 arguments。

## 未复制的内容

上游 `../claimarc/data/cache` 约 436 GB，是 LLM/VLM 请求缓存，不属于论文中的原始观测、正式中间表或最终训练数据，因此未复制。它可以减少重新调用成本，但不是从 raw 到 final 的必要科学数据。若审稿要求逐次复放外部模型输出，应另做去敏、版本化的 cache release，并记录供应商模型 revision 与调用时间。

## 数据发布注意

raw 层可能包含用户评论、直播文本和商品图片。对外提交前需确认授权、平台条款、个人信息去标识化和图片版权。复现包完整保存不等于可以公开分发；若不能公开，应提供受控访问、脱敏版本或数据使用申请流程，并保持相同 pair_id/hash 映射。
