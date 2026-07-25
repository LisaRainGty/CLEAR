# 公平比较协议

## 冻结口径

- 数据：`data/dataset_duallabel_FULLPOOL_PLUS_OBJNEG_supervised_20260615.jsonl`
- SHA-256：`1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76`
- 标签：`y`；可靠性：`c`
- 划分：已有 `split`；room_id 分组；不得重新随机切分
- 证据视图：`sources_only`，即按 `[PARAM]` → `[OCR]` → `[VLM]` 顺序拼接每条样本实际存在的来源；不按来源数过滤样本
- 禁止输入：不读取、不生成 `arguments`；不运行 params-only/OCR-only/VLM-only 实验
- seeds：0、1、2
- 阈值：只在 validation 以 Macro-F1 选择；原样用于 test
- checkpoint/超参/输入策略：只允许 validation 选择
- 主指标：AP；同时报告 AUC、Accuracy、F1_pos、Macro-F1、wF1、ECE
- 显著性：保存逐样本概率后，在相同 test pair 上做 2,000 次配对 bootstrap

## 为什么不能让每个模型直接采用“测试集上最好数据”

公平不等于“给每个模型挑一个测试集最高分的视图”。那会对测试集重复选择，使结果乐观且不可复现。允许的两种设计是：

1. **共享输入主表**：所有模型使用同一 `sources_only` 视图。这是论文主张模型结构/学习目标优越性时最干净的比较。
2. **验证集调优表**：预先定义每个模型相同的候选视图集合与预算，每个模型仅根据 validation 选择视图，冻结后评估 test。该表必须另列为 “validation-tuned systems”，不能替代共享输入主表。

当前 full model 的历史 `args_only` 不是“最好数据”：冻结数据中 arguments 全空，它实际是空 evidence 诊断；而且该策略曾根据 test AP 被提升为 canonical。本复现包已将该视图从任务矩阵中完全移除。

## 相同数据与模型容量

每个模型收到完全相同的原始 claim 和三源拼接字符串。CLAIMARC/ESIM/DAM 的双流结构对 claim 与 evidence 分别使用 384-token 上限；BERT/RoBERTa/TextCNN/BiLSTM 单序列模型使用其预先声明的 512-token 上限。这一差别是模型架构容量的一部分，不是数据视图选择；所有截断在不查看测试结果前冻结并在运行元数据中报告。

## 逐表重跑规则

- Table 3：全部可学习模型三种子；冻结探针一次但固定随机状态；LLM 保存原始响应与解析结果。
- Table 4：每个留一品类折和每个留主播 seed 使用同一 evidence policy；保存逐折预测与目标域定义。
- Table 5：注入样本与评估样本必须 disjoint；每个比例的抽样 seed 固定并记录。
- Table 6：三变体除 `cl_mode` 外完全相同；同一数据、骨干、训练步数和 seeds。
- Table 7–10：每行只改变表中命名的一个因素；canonical 必须来自同一 runner 和输入视图。
- Table 12：22 个 LoRA 设置只与 LoRA canonical seed 0 比；删除原草稿中 3 个单源输入行，full-FT 下不报告 LoRA rank 灵敏度。
- Table 13：全部公式扰动与 canonical seed 0 直接匹配，不再用单种子扰动对比三种子均值。
- Figures 1、6、7：从公平主表同一预测 bundle 重新生成；不得拼接不同 canonical。

## 每次运行必须保存

结果行和预测 bundle 至少包含：dataset path/hash、label field、split/group field、evidence policy、模型真实解析路径或 revision、seed、全部超参、验证阈值、val/test pair_id、概率、标签、可靠性、开始/结束时间、运行命令、代码提交号和硬件/库版本。

`configs/paper_fair.json` 是当前冻结协议；`scripts/run_paper_suite.py` 是统一入口。任何偏离都应生成新的配置文件，而不是覆盖现有文件。
