# data/ — 数据集说明与来源

## 最终数据集

- **文件名**：`dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl`
- **规模**：4,883 对（streamer-claim × product-fact），覆盖 10 个一级品类、38 个子品类、108 个直播间、1,532 个归一化属性。
- **正类（误导）占比**：25.6%（类不平衡的筛查任务）。
- **样本构成**：评论驱动样本 2,278 条；客观负例（objective-negative）2,605 条。
- **证据来源覆盖**（0/1/2/3 个产品事实来源）：1,483 / 1,826 / 1,118 / 456，均值 1.11。
- **实例可靠性权重 `c_{p,a}`**：均值 0.46，中位数 0.48，主体区间 [0.18, 0.82]。

## 划分（按 room_id 分组，70:10:20，确保同一主播不跨划分）

| Split | #Pairs | Pos.% | #Rooms |
|---|---:|---:|---:|
| Train | 3,636 | 24.2 | 92 |
| Val   | 392   | 31.1 | 5  |
| Test  | 855   | 29.1 | 11 |
| **All** | **4,883** | **25.6** | **108** |

## 文件位置（已包含在本资料包内）

```
data/final/repaired_v1/
├── dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl   # 最终监督数据集（4,883 行，18M）
├── dataset_planbaseline_duallabel_all_20260614.jsonl                  # 合并客观负例前的基础对偶标签集（构建输入）
└── stateful_proposal_dataset_v2_vlm120_plus_bigexpand_all_20260614.jsonl  # 评论驱动样本池（更上游输入）
```

代码默认读取路径为 `data/final/`；运行训练/评估时把最终数据集放在 `data/final/`（或用 `--dataset` 显式指向 `data/final/repaired_v1/...`）即可。

### 关键字段
`pair_id, product_id, room_id, category, subcategory, attribute_id, attribute_name, product_title, claim,`
`y`（感知风险标签，正类=误导）, `y_perception`, `label_observed`, `c` / `c_reliability`（实例可靠性权重）,
`sample_role, contrastive_mask`（对比学习掩码）, `evidence_params / evidence_ocr / evidence_vlm`（三路产品事实证据）,
`coverage, evidence_count, split`（train/val/test 划分）, 以及若干审计与重建辅助字段。

## 构建流程（溯源）

最终数据集由“评论驱动的感知风险样本池（FULLPOOL）”与“客观负例池（OBJNEG）”合并、并附加实例级可靠性权重而成。相关脚本见 `../code/data_build/`：
- `build_stateful_proposal_dataset_v2.py` — 主样本池构建；
- `build_objective_negative_dataset_v1.py` — 客观负例构建与合并；
- `build_plan_label_weights_v1.py` — 标签与可靠性权重。

逐步构建/审计的状态化文档见 `../docs/dataset_provenance/`。

## 备注

本地另存有更早一批不同配置的数据集族（如 `dataset.jsonl`、`dataset_hq_broad_*` 等）与对应预测包，与论文最终口径（N=855/249）不同，未纳入本资料包以免混淆。
