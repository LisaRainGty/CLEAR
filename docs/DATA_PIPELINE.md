# CLAIMARC 数据流水线（从原始数据到最终训练集）

本文件给出从**原始直播/电商数据**到论文最终训练集
`dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl` 的**完整、可溯源**
构建链路：每一阶段的输入/输出、对应代码、产物落盘位置,以及哪些环节依赖 LLM/VLM。

> 设计原则(见 `Methodology_Data.md` §2):**客观商品证据的矛盾本身不直接生成正标签**;
> 缺证据/缺话术转为"修复状态"或低可靠性 silver,而非强制负例。消费者感知(对齐评论是否
> 反驳被修复后的主播话术)决定 `y_perception`,可靠性 `c` 由 §2 的 f_sat/f_cov/f_asym 等聚合。

## 0. 目录与运行约定

```
data/
├── raw/                      # 原始数据(本地;不入库)
│   ├── comment/              #   评论(≈28,556 条)
│   ├── srt_cut/              #   直播切片字幕(主播话术来源)
│   └── product_images/       #   商品详情图(Stage C 视觉证据)
├── processed/                # 各阶段中间产物(本地;不入库)
│   ├── stageA/  stageB/  stageC/  ...
│   └── labels.jsonl
├── index/                    # 检索/商品索引(本地;不入库)
└── final/                    # 数据集与其变体(本地;不入库)
    └── repaired_v1/          #   重建/修复链路产物 + 最终训练集
```

所有命令从 `src/` 运行(`PYTHONPATH=src`,见根 `README.md`/`env.sh`):
```bash
source env.sh && cd src
```
全量编排入口:`python -m run_pipeline --all`(也支持 `--pilot` 小样、`--stage A0 A1 --category food_and_beverages` 局部跑)。
阶段顺序见 `run_pipeline.py`:`A0→A1→A2→A3→B0→B1→B2B3→B4B5→C1→C2→C3→C4→C5→labels→final`。

> ⚠️ 真实从零复现需要:原始数据(≈34GB)、可用的 LLM/VLM 网关(`common/llm.py`,经
> `MATPOOL_API_KEY` 等环境变量配置)与 GPU。A1/B1/C4 等阶段含大量 LLM/VLM 调用,
> 因此该链路的"完全重跑"成本很高;`docs/dataset_provenance/` 完整记录了每个数据集
> 变体的输入清单与统计,作为不可逐字节重放环节的**溯源证据**。

---

## 1. Stage A — 评论属性抽取与品类内标准化  (`src/stage_a/`)

把每条评论的属性提及映射到**品类内标准化**的 `attribute_id`,为 (product, attribute)
粒度聚合提供干净 aspect 流。

| 步 | 模块 | 输入 | 输出 |
| --- | --- | --- | --- |
| A0 | `stage_a.a0_build_cas` | 商品"产品参数"字段 | `CAS_{cat}.json`(品类属性表) |
| A1 | `stage_a.a1_extract_aspects` | 评论 + CAS（**LLM**） | `raw_aspects.jsonl` |
| A2 | `stage_a.a2_aggregate_free` | `FREE::` 类 aspect | `CAS+_{cat}.json` |
| A3 | `stage_a.a3_resolve_labels` | `FREE::` → 标准 id | `resolved_aspects.jsonl` |

落盘:`data/processed/stageA/`。细节(CAS 结构、prompt、polarity/mention_strength 等
字段语义)见 `Methodology_Data.md` §1 Stage A。

## 2. Stage B — 话术抽取、段落化与三元组对齐  (`src/stage_b/`)

从直播字幕抽取主播话术(claim),并与 (product, attribute) 对齐成三元组。

| 步 | 模块 | 作用 |
| --- | --- | --- |
| B0 | `stage_b.b0_acmt` | 构建 ACMT(attribute–claim 映射表) |
| B1 | `stage_b.b1_claim_extract` | 从 `srt_cut/` 抽取主播话术（**LLM**） |
| B2B3 | `stage_b.b2_b3_passage` | 话术段落化 / passage 组织 |
| B4B5 | `stage_b.b4_b5_align` | 话术 ↔ 商品属性 ↔ 评论 三元组对齐 |

落盘:`data/processed/stageB/`(及 product_v2 等变体)。

## 3. Stage C — 商品视觉/参数证据  (`src/stage_c/`)

从商品详情图与参数构建**客观商品证据**(fact records)。

| 步 | 模块 | 作用 |
| --- | --- | --- |
| C1 | `stage_c.c1_image_triage` | 详情图初筛/分类 |
| C2 | `stage_c.c2_params` | 结构化参数抽取 |
| C3 | `stage_c.c3_ocr` | 详情图 OCR |
| C4 | `stage_c.c4_vlm` | 视觉语言模型读图取证（**VLM**） |
| C5 | `stage_c.c5_fact_records` | 汇总为 `fact_records`(每属性证据 + coverage) |

落盘:`data/processed/stageC/`(负例证据在 `stageC_neg/`)。

## 4. 标签引擎与基础数据集  (`src/labels/`, `src/final/`)

| 模块 | 作用 | 输出 |
| --- | --- | --- |
| `labels.build_labels` | §2 硬标签 `y` 与可靠性权重 `c`(f_sat/f_cov/f_asym/f_fake + 证据分 w_i) | `data/processed/labels.jsonl` |
| `final.join_split` | 合并 A/B/C + 标签,按**直播间分组**划分 train/val/test(防泄漏) | `data/final/dataset.jsonl` |

## 5. 重建/修复与最终训练集  (`src/data_quality/`)

在基础数据集之上,经"全对重建 + 多轮 LLM/VLM 评审"得到 proposal-faithful 的
双标签数据集,再拼接客观负例,形成论文最终训练集。链路(产物落 `data/final/repaired_v1/`):

```
full_pair_reconstruction_queue_v1            (重建队列)
        │  +  多轮 LLM/VLM 评审产物(见 dataset_provenance/*.md 的 Inputs 清单)
        ▼
rebuild_repaired_datasets_v1 / build_full_pair_promoted_dataset_v1
        ▼
build_stateful_proposal_dataset_v2           →  stateful_proposal_dataset_v2_FULLPOOL_{all,supervised,contrastive,repair}
        │     (分离 y_perception / promotion_state / c_reliability / contrastive_mask)
        ▼
build_plan_label_weights_v1                  →  dataset_planbaseline_duallabel_FULLPOOL_{supervised,all}
        │     (§2 y/c 标签引擎,纯离线聚合,无 LLM)
        │
build_objective_negative_dataset_v1          →  dataset_objective_negatives_v1_20260615
        │     (claim-without-comment 的 y=0 证据驱动负例;coverage→置信度,PU 折扣 κ)
        ▼
   concat(plan-baseline FULLPOOL supervised  +  objective negatives)
        ▼
dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl   ← 论文最终训练集
```

各脚本的精确输入/输出路径见其 `argparse` 默认值;每个数据集变体的输入清单与统计
(reviewed/observed/contrastive/repair 行数、`sample_role`/`promotion_state` 分布、
split 与泄漏检查)见 `docs/dataset_provenance/STATEFUL_PROPOSAL_DATASET_V2_*.md`
(总览以 `..._FULLPOOL_20260614.md` 为准)。质量自检:`data_quality.audit_dataset_quality`。

> 最终拼接为一步 `cat`(plan-baseline FULLPOOL supervised 行 + objective negatives 行),
> 故文件名后缀 **`PLUS_OBJNEG`**。最终训练集随后置于 `data/`(模型代码读取 `data/dataset_*.jsonl`)。

## 6. 字段说明

最终训练集逐字段语义见 `data/README.md` 与 `docs/DATA_README_fields.md`。关键字段:
`y`(客观核验/感知硬标签)、`y_perception`、`c`/`c_reliability`(样本可靠性权重)、
`sample_role`、`contrastive_mask`、`claim`(主播话术 + 带时间戳 segments)、
`evidence_params`/`evidence_ocr`(商品证据)、`arguments`(无标签泄漏的支持/反驳论证)。

## 7. 从原始数据复现(摘要)

```bash
source env.sh && cd src
# (a) raw -> 结构化记录 -> 基础数据集(含 LLM/VLM 阶段,需网关与 GPU)
python -m run_pipeline --all
# 小样冒烟:python -m run_pipeline --pilot
# (b) 记录 -> 最终监督数据集(在 data/final/repaired_v1/ 产物之上)
python -m data_quality.build_stateful_proposal_dataset_v2
python -m data_quality.build_plan_label_weights_v1
python -m data_quality.build_objective_negative_dataset_v1
# (c) 拼接最终训练集(plan-baseline FULLPOOL supervised + objective negatives)
cat data/final/repaired_v1/dataset_planbaseline_duallabel_FULLPOOL_supervised_*.jsonl \
    data/final/repaired_v1/dataset_objective_negatives_v1_20260615.jsonl \
    > data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl
# (d) 质量自检
python -m data_quality.audit_dataset_quality
```

> 不可逐字节重放的环节(LLM/VLM 评审)以 `docs/dataset_provenance/` 的输入清单 + 统计
> 作为溯源证据;模型侧复现见根 `README.md`。
