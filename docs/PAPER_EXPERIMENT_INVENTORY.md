# 论文实验总表

完整可读正文位于 `paper/draft/实验结果与分析_完整提取.md`；该文件连同 `media/` 保留了草稿内全部文字、表格和七张嵌入图片。下面按论文研究问题汇总全部实验、当前数值和可追溯状态。

## RQ1：主分布比较

Table 3 在测试集 N=855（249 正例）比较 14 条基线与 CLAIMARC。论文数值为：

| 组别 | 模型 | Acc | F1 | AP | AUC |
|---|---|---:|---:|---:|---:|
| 客观核验 | ESIM | 73.8 | 65.9 | 51.2 | 79.9 |
|  | Decomposable Attention | 76.2 | 61.5 | 60.5 | 83.6 |
|  | BERT-NLI | 79.1 | 65.1 | 67.9 | 87.4 |
| 文本分类 | TextCNN | 76.5 | 57.0 | 60.9 | 84.1 |
|  | BiLSTM | 79.2 | 65.5 | 66.3 | 86.4 |
|  | BERT-CLS | 80.9 | 71.7 | 68.2 | 88.3 |
|  | RoBERTa-CLS | 80.5 | 70.0 | 69.0 | 88.4 |
| 冻结探针 | BGE + LR / SVM / MLP / kNN | 76.0–78.6 | 62.1–68.5 | 60.5–67.6 | 83.0–86.6 |
| LLM | Qwen-Flash 0-shot | 49.8 | 30.5 | 27.4 | 46.1 |
|  | GPT-5.4 5-shot CoT | 58.2 | 24.8 | 26.9 | 44.6 |
|  | Qwen2.5-7B LoRA SFT | 73.0 | 61.7 | 63.8 | 81.9 |
| 本文 | CLAIMARC | **82.6** | **73.4** | **75.4** | **90.4** |

Figure 1 展示 PR/ROC 曲线。这些数字是草稿历史值：CLAIMARC 行来自空 argument view，BERT-NLI 可能回退。公平重跑已补齐 Qwen2.5-7B QLoRA SFT 代码并禁止 NLI 静默回退；最终只采用新任务产物。

## RQ3：跨域与免梯度检索库适配

Table 4：留一品类 10 折与留 20 主播 3 seeds。论文历史报告 CLAIMARC 为 AUC/AP = 90.0±4.0 / 71.0±5.5，以及 93.1±0.3 / 81.4±1.3。历史聚合文件因逐样本 bundle 不完整且 source-view 不一致，已从发布树移出。新跨域产物要求记录每折 pair-id hash、room 划分与交集检查，并且聚合器拒绝缺少这些 provenance 的 bundle。

Table 5：冻结前向分类器 AP/AUC/F1 = 81.1/93.1/73.5；检索库从 source-only 的 65.5/80.3/72.4 随目标池 20/40/60/80/100% 注入，最终升至 71.5/84.8/74.9。Figure 3 是相同注入曲线。需要恢复或重跑逐折 bundle，验证“写入池”和“读取集”始终不相交。

## RQ2：表征几何

Table 6 比较相同 BGE full-FT 骨干下 w/o contrast、SupCon、RACL：

| 变体 | Silhouette | Hard purity@10 | Alignment | Uniformity |
|---|---:|---:|---:|---:|
| w/o contrast | .196±.053 | .500±.043 | 1.121 | -2.02 |
| SupCon | .349±.173 | .573±.017 | .478 | -.86 |
| RACL | **.422±.010** | .573±.088 | .905 | -1.21 |

Figure 4 为 UMAP。新结果只从 `embeddings/fair_rerun/emb_geom/` 的三种子嵌入计算，表格写入 `results/fair_rerun/table6_geometry.json`，图写入 `paper/figs/fig_umap_label.*`。

## RQ4：组件、结构、检索策略与稳健性

- Table 7：去 RACL、去可靠性权重、去类平衡、去 ESIM 四元组、BGE→BERT。论文 canonical AP 75.4，各消融 AP 72.1/72.2/73.9/71.8/71.1。
- Table 8：双流 canonical、去 fusion、claim-only、evidence-only，论文 AP 75.4/73.8/70.4/48.4。
- Table 9：hard positive、same-attribute/same-evidence negatives、Kp=1/5、Kn=1/10，论文 AP 范围 73.5–74.0，对 canonical 75.4。
- Table 10：uniform/reversed/permuted/binary/count-only/sqrt(c) 可靠性权重，论文 AP 72.2/74.1/73.0/68.4/73.1/72.6，对 canonical 75.4。
- Figure 5 与 Table 12：原草稿声称 26 个单种子超参/结构/证据视图设置，且将 LoRA-frozen 扫描与 full-FT 三种子 canonical 混用。新表固定为 22 个全三源 LoRA seed-0 设置，用同块中的 LoRA canonical 作参照；params-only/OCR-only/VLM-only 三行已从任务矩阵删除。
- Table 13：c 公式的 k、lambda、rho、phi 单种子扰动；论文 AP 从 .678 到 .726，对三种子 canonical .754。应改为同一 seed 对同一 seed，或每个设置全部三种子。
- Figure 6：按 forward–RKC 分歧弃判；100% 覆盖 Acc/AP 约 .814/.737，80% 为 .867/.778，70% AP .809，65% AP .835。
- Figure 7：temperature scaling 后可靠性图。错误分析报告 TP=210、FN=39、FP=120；按证据源 1/2/3 的错误率为 24.1%/16.4%/10.3%。

论文正文从 Table 10 直接跳到 Table 12，没有 Table 11；投稿前应补表或重编号。配对 bootstrap 固定 n=2,000，仅对完成且通过 pair-id 顺序检查的新预浏 bundle 执行；冻结探针按设计使用 seed 0，可学习模型使用三种子概率集成。

## 数据描述表

Table 1、Table 2 的样本规模、划分、正例率、证据覆盖、可靠性和品类计数均已从最终 JSONL 复核。唯一待澄清的描述统计是“商品事实文本约 22 字”；当前原始三源文本按 pair 合计均值约 51 字。

## 图与结果文件映射

| 论文对象 | 当前文件 |
|---|---|
| Figure 1 PR/ROC | `paper/figs/fig_pr_roc.*`, `results/fair_rerun/metrics_rich.json` |
| Figure 3 注入曲线 | `paper/figs/fig_inject.*`, `results/fair_rerun/table5_injection_rooms.json` |
| Figure 4 几何/UMAP | `paper/figs/fig_umap_label.*`, `results/fair_rerun/table6_geometry.json` |
| Figure 5 超参 | `paper/figs/fig_hparam.*`, `results/fair_rerun/paper_tables.json` |
| Figure 6 选择性预测 | `paper/figs/fig_selective.*`, `results/fair_rerun/selective_canon.json` |
| Figure 7 校准 | `paper/figs/fig_calibration.*`, `results/fair_rerun/metrics_rich.json` |
| 错误分析 | `results/fair_rerun/error_analysis.json` |

发布版结果统一位于 `results/fair_rerun/`、逐样本 bundle 位于 `embeddings/fair_rerun/`。历史异口径文件已移至 `main` 外的可恢复归档，不作为最终论文结果输入。
