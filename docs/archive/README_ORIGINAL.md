# CLAIMARC — 直播电商虚假宣传识别的检索增强对比学习框架（最终版）

本仓库收录 **CLAIMARC 最终架构**（**不使用属性分块**、**BGE 全量微调** + 类别均衡监督
对比）的全部可复现资产：核心代码、最终数据集、实验结果、嵌入库与论文（中英双版）。
组织目标是"打开本文件夹即可继续修改、并在新环境中快速复现"。

> **Canonical（论文最终方法，经确认）**：BGE 全量微调（`--enc_train full --lr 1e-5`）
> + 全池检索增强对比 RACL，**正/负例在全局标签池内挑选、不做属性分块**、类别均衡
> （`--cl_mode racl`，等价于 `--cl_no_attr_block --cl_class_balanced`）。
> 论文头条指标 **Acc 82.6 / F1 73.4 / AP 75.4 / AUC 90.4**。
>
> 推理主路径用前向分类头（CLS）；RKC（检索近邻投票）与 CLS 配合，用于选择性预测
> 的人工 abstain 门控与免梯度新域吸纳。

## 目录结构

```
claimarc_final/
├── README.md                 # 本说明
├── requirements.txt          # 依赖（含版本）
├── docs/archive/requirements_dev_snapshot.txt # 原开发环境 pip freeze
├── env.example.sh            # 环境变量样例（复制为 env.sh；设 PYTHONPATH/可选本地 BGE 路径）
├── src/
│   ├── config.py             # 路径 / 模型 / 网关配置（全部读环境变量，无硬编码密钥）
│   ├── common/               # LLM 网关、embedding、IO 等公用模块
│   ├── run_pipeline.py       # 数据流水线编排器（A0→…→final）
│   ├── stage_a/ stage_b/ stage_c/   # 评论属性 / 话术对齐 / 商品证据 三阶段
│   ├── labels/ final/        # §2 标签引擎 + 分组划分 join
│   ├── data_quality/         # 数据集重建/拼接（records→最终训练集）
│   └── models/               # 建模与实验核心代码（见下方“代码地图”）
├── docs/                     # 数据流水线与溯源文档
│   ├── DATA_PIPELINE.md      # 从原始数据到最终训练集的完整 DAG（必读）
│   ├── Methodology_Data.md   # §2 标签方法学
│   └── dataset_provenance/   # 每个数据集变体的输入清单与统计（溯源）
├── data/                     # 数据（本地保留，.gitignore 不入库；见各层 README）
│   ├── raw/ processed/ index/ final/   # 原始 / 中间 / 索引 / 数据集变体
├── results/                  # 实验结果（入库）
│   ├── campaign6_results.jsonl   # RQ1 主表 + 主消融（新 canonical 重定基）
│   ├── campaign7_results.jsonl   # 决策路径 + RACL 细粒度消融
│   ├── campaign8_results.jsonl   # 双流 / 分类头 / 骨干消融
│   ├── alpha_sweep.json          # CLS×RKC 融合系数扫描
│   └── artifacts/                # 图表与案例分析的中间 JSON（含 RQ2 geom2.json）
├── embeddings/               # 嵌入库 *.pt（本地保留，.gitignore 不入库；见 embeddings/README.md）
└── paper/                    # 论文（中英双版）+ 图
    ├── claimarc_paper.tex / .pdf     # 英文 LaTeX + 编译 PDF（latexmk -xelatex）
    ├── claimarc_paper_zh.md / .docx  # 中文 Markdown + DOCX（pandoc）
    ├── RESULTS_LOG.md                # 完整实验结果记录（口径与数值）
    └── figs/                         # 论文图（pdf + png）
```

## 代码地图（`src/models/`）

按论文的研究问题（RQ）组织；从 `src/` 目录以 `python -m models.<name>` 运行。

**核心库（被各入口复用）**
| 文件 | 作用 |
| --- | --- |
| `model.py` | CLAIMARC 模型：双流编码 + `TwoStreamFusion` + RACL + 可靠性加权 |
| `data.py` | 数据加载 / 分词 / 划分；`resolve_bge_path` 支持本地目录或 `CLAIMARC_BGE_PATH` |
| `train.py` | 训练 / 评估 / RKC / 阈值搜索（主入口，CLI）。`--cl_mode {racl,supcon,none}` 选择对比目标 |
| `metrics_rich.py` | 富指标汇总 + 主分布 PR/ROC 图 `fig_pr_roc` |
| `compile_results.py` | 汇编各 campaign 的结果表 |

**RQ1 主比较 + 基线**
| 文件 | 作用 |
| --- | --- |
| `run_campaign6.py` | RQ1 主表 + 主消融的批处理编排（新 canonical 重定基） |
| `baselines.py` / `baselines_ft.py` / `baselines_frozen.py` / `baselines_neural.py` | 冻结探针 / 微调 / 神经基线 |
| `run_llm_baselines.py` | LLM 基线（经 `common.llm` 网关） |

**RQ2 表征几何（§4.6，去属性、含 SupCon-2020 对照）**
| 文件 | 作用 |
| --- | --- |
| `run_geom_campaign.py` | 受控三变体 ×3 种子：`none` / `supcon`(Khosla 2020) / `racl`，全量微调并导出 train+test 嵌入 |
| `geom_probe2.py` | 标签条件几何：silhouette、alignment/uniformity、kNN purity@k、困难区拆分（**无属性条件**） |
| `make_geom_figs.py` | RQ2 图：`fig_umap_label`(3 面板) / `fig_geometry` / `fig_knn_purity` |

**RQ3 跨域泛化与免梯度库吸纳**
| 文件 | 作用 |
| --- | --- |
| `crossdomain.py` | 跨域评估（留一类 / 留主播） |
| `xdom_common.py` / `xdom_fold.py` / `xdom_inject.py` / `xdom_agg.py` / `xdom_llm.py` | 跨域协议、检索库注入、聚合、LLM 跨域 |
| `make_inject_fig.py` | 库注入轨迹图 `fig_inject` |

**RQ4 消融与超参 / 选择性预测 / 案例**
| 文件 | 作用 |
| --- | --- |
| `run_campaign7.py` / `run_campaign8.py` | 决策路径、RACL 细粒度、双流 / 分类头 / 骨干消融 |
| `alpha_sweep.py` | CLS×RKC 融合系数扫描（`fig_hparam` 之一） |
| `make_selective_fig.py` | 选择性预测风险–覆盖曲线 `fig_selective` |
| `make_figs.py` | `fig_calibration`（校准）/ `fig_hparam`（超参敏感性）（RQ2 几何图见 `make_geom_figs.py`，PR/ROC 见 `metrics_rich.py`） |
| `case_confounder.py` / `case_counterfactual.py` | 误差 / 反事实案例分析 |

> 说明：`train.py` 仍保留属性分块负例的代码路径（`--cl_attr_block` 等），**仅用于复现
> “弃用属性分块”的那一行消融对照**；canonical 方法不启用它。

## 环境准备（新机器从零搭建）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 论文训练请改用 requirements-train.txt
cp env.example.sh env.sh                 # 设 PYTHONPATH=src；可选填本地 BGE 路径与 LLM 网关
source env.sh
```

主要依赖：PyTorch 2.6（CUDA 12.x）、transformers 4.57、sentence-transformers、
scikit-learn、umap-learn、matplotlib。训练在单卡 RTX 4090 上进行。
首次运行会自动从 ModelScope/HuggingFace 拉取 `BAAI/bge-large-zh-v1.5`；若已本地下载，
可在 `env.sh` 中设 `CLAIMARC_BGE_PATH=/path/to/bge-large-zh` 直接离线加载。

## 快速复现

均从 `src/` 目录运行（`PYTHONPATH` 已含 `src`）。数据集见 `data/`。

**1) 训练 canonical（论文最终方法）**
```bash
cd src
python -m models.train \
  --dataset ../data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl \
  --encoder_name BAAI/bge-large-zh-v1.5 \
  --enc_train full --lr 1e-5 \
  --cl_mode racl --cl_no_attr_block --cl_class_balanced \
  --tau 0.07 --lambda_cl 0.5 --Kp 3 --Kn 5 --loss bce \
  --warmup 3 --cl_epochs 6 --seed 0
```
> `--cl_no_attr_block --cl_class_balanced` 是 canonical 的关键：正/负例在全局标签池内
> 挑选、不做属性分块。省略它们会回退到属性分块变体（仅用于消融对照）。

**2) RQ1 / RQ3 / RQ4 各表**
```bash
python -m models.run_campaign6     # RQ1 主表 + 主消融
python -m models.run_campaign7     # 决策路径 + RACL 细粒度消融
python -m models.run_campaign8     # 双流 / 分类头 / 骨干消融
python -m models.alpha_sweep       # CLS×RKC 融合扫描
```

**3) RQ2 表征几何（§4.6，三变体对照）**
```bash
python -m models.run_geom_campaign \
  --dataset ../data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl \
  --outdir ../embeddings --seeds 0 1 2
python -m models.geom_probe2 \
  --emb_dir ../embeddings --seeds 0 1 2 --out ../results/artifacts/geom2.json
python -m models.make_geom_figs \
  --emb_dir ../embeddings --geom_json ../results/artifacts/geom2.json --outdir ../paper/figs
```

**4) 重绘其余论文图**
```bash
python -m models.metrics_rich        # fig_pr_roc（主分布 PR/ROC）
python -m models.make_figs           # fig_calibration / fig_hparam
python -m models.make_inject_fig     # fig_inject（跨域库注入）
python -m models.make_selective_fig  # fig_selective（选择性预测）
```

## 数据流水线（从原始数据复现）

完整链路(原始数据 → 三阶段抽取 → 标签引擎 → 重建/拼接 → 最终训练集)、每阶段的
脚本↔产物映射、以及哪些环节依赖 LLM/VLM,见 **[`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md)**。

```bash
source env.sh && cd src
python -m run_pipeline --all        # raw -> 结构化记录 -> 基础数据集（含 LLM/VLM 阶段）
python -m run_pipeline --pilot      # food_and_beverages 小样冒烟
# records -> 最终监督数据集
python -m data_quality.build_stateful_proposal_dataset_v2
python -m data_quality.build_plan_label_weights_v1
python -m data_quality.build_objective_negative_dataset_v1
```

> 原始数据(≈34GB)、processed(≈6.4GB)、final(≈8.5GB)均**本地保留、不入库**;在同一块盘上
> 用 `mv ../claimarc/data/<layer>/* data/<layer>/` 迁入即可(瞬时、零额外占用)。各层用途见
> `data/<layer>/README.md`。从零重跑需 LLM/VLM 网关与 GPU;不可逐字节重放的评审环节以
> `docs/dataset_provenance/` 的输入清单 + 统计作为溯源证据。

### 数据流水线代码（`src/`）
| 模块 | 作用 |
| --- | --- |
| `run_pipeline.py` | 按依赖顺序编排全部阶段（`--all` / `--pilot` / `--stage`） |
| `stage_a/`（a0–a3） | 评论属性抽取 + 品类内标准化（A1 用 LLM） |
| `stage_b/`（b0–b4_b5） | 直播话术抽取 + 段落化 + 三元组对齐（B1 用 LLM） |
| `stage_c/`（c1–c5） | 商品详情图/参数证据：triage/params/OCR/VLM/fact records（C4 用 VLM） |
| `labels/build_labels.py` | §2 硬标签 `y` 与可靠性权重 `c` |
| `final/join_split.py` | 合并三阶段 + 直播间分组划分（防泄漏） |
| `data_quality/build_stateful_proposal_dataset_v2.py` | 重建评审 → 双标签 stateful 数据集 |
| `data_quality/build_plan_label_weights_v1.py` | FULLPOOL 上的 §2 y/c 标签引擎（纯离线） |
| `data_quality/build_objective_negative_dataset_v1.py` | 证据驱动客观负例（y=0，PU 折扣） |
| `data_quality/audit_dataset_quality.py` | 数据集质量自检 |

## 数据与嵌入说明

按隐私要求，**最终数据集（直播话术 / 评论文本，约 56MB）与嵌入库 `*.pt` 不入 git**，
仅在本文件夹本地保留。字段说明见 [`data/README.md`](data/README.md)；嵌入清单见
[`embeddings/README.md`](embeddings/README.md)。`results/` 与 `paper/` 下的结果 JSON、
图与论文源文件均已入库，可直接据此校对数值与重绘图。

## 论文

- 英文：`paper/claimarc_paper.tex` →（`latexmk -xelatex`）`paper/claimarc_paper.pdf`
- 中文：`paper/claimarc_paper_zh.md` →（`pandoc`）`paper/claimarc_paper_zh.docx`
- 完整数值口径见 `paper/RESULTS_LOG.md`
