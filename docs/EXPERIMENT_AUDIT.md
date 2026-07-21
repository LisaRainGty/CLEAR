# 实验与结果审计

审计对象是论文草稿 `paper/draft/实验结果与分析.docx`、最终数据、训练代码、历史结果以及论文 `RESULTS_LOG.md`。结论是：数据构造主干可以复核，但当前论文结果表还不能证明“所有模型在相同数据与输入预算上公平比较”。

## 一、已经核实的内容

| 项目 | 论文口径 | 数据核验 | 状态 |
|---|---:|---:|---|
| 总 pair 数 | 4,883 | 4,883 | 一致 |
| train / val / test | 3,636 / 392 / 855 | 3,636 / 392 / 855 | 一致 |
| 三划分正例 | 879 / 122 / 249 | 879 / 122 / 249 | 一致 |
| 总正例率 | 25.6% | 25.60% | 一致 |
| 一级品类 / 二级品类 / 直播间 / 属性 | 10 / 38 / 108 / 1,532 | 一致 | 一致 |
| 评论驱动 / 客观负例 | 2,278 / 2,605 | 一致 | 一致 |
| 证据覆盖 0/1/2/3 | 1,483 / 1,826 / 1,118 / 456 | 一致 | 一致 |
| 可靠性 c | 均值 .456，中位数 .48，范围 .18–.82 | 一致 | 一致 |
| room 跨划分泄漏 | 无 | train/val/test 交集均为空 | 一致 |
| arguments 覆盖 | 未在实验正文披露 | 0 / 4,883 | 关键缺口 |

主播 claim 片段的字符均值约 22.96，与论文“约 23 字”一致。当前代码对 params/OCR/VLM 原始文本直接求和时，商品事实均值约 51 字，不是论文写的“约 22 字”；需确认论文统计的是截断文本、单证据均值还是其他字段，并把统计脚本纳入复现包。

## 二、头条结果的实际来源

论文头条的 Accuracy / F1_pos / AP / AUC = **0.8261 / 0.7335 / 0.7541 / 0.9042**，与 `results/campaign6_results.jsonl` 中 `c6_args` 三种子的均值吻合。

`c6_args` 使用 `--evidence_policy args_only`。然而当前最终训练文件 4,883 条记录没有一条包含非空 supporting/refuting/gap argument。因此其 evidence 流实际上只保留属性名和结构标记，没有商品事实正文。这个运行可以作为“无事实证据/空 evidence”诊断，不能命名为完整模型的最佳数据，也不能替换主比较 canonical。

同一结果文件中的 `c6_canon` 使用默认 `args_first`。由于 arguments 为空，它实际退化为 params/OCR/VLM sources，与多数基线使用的输入相近。三种子均值为：

| 历史标签 | Evidence 实际内容 | Acc | F1_pos | AP | AUC |
|---|---|---:|---:|---:|---:|
| `c6_args` | 空 arguments；无商品事实正文 | .8261 | .7335 | .7541 | .9042 |
| `c6_canon` | params + OCR + VLM | .8238 | .7378 | .7363 | .9000 |

差异不代表 `args_only` 是“更好的数据”；它更可能说明属性名、标签结构或数据偏差已足以形成预测信号。选择 AP 最高的测试运行作为 canonical 还构成测试集调参。

## 三、哪些论文表格混用了口径

论文 Table 7–10 和 Table 13 的 canonical 行使用 `c6_args`，而被比较的消融来自 `c6_canon`、campaign7、campaign8、campaign5 或 c-formula 重跑，通常读取 source evidence。这样“只改变一个因素”的前提不成立。

- Table 7：`c6_no_cl`、`c6_no_weight`、`c8_concat`、`c6_bert` 对 `c6_args`。
- Table 8：campaign7/8 的 stream/fusion 变体对 `c6_args`。
- Table 9：campaign7 的 RACL 挖掘变体对 `c6_args`。
- Table 10：历史可靠性权重变体对 `c6_args`。
- Table 13：单种子 c 重算变体对三种子 `c6_args`，同时混合输入视图与种子口径。
- Table 6 与跨域实验使用单独的 source-view 训练，不能用 `c6_args` 作为同口径 in-domain 参照。
- Figure 6 的 100% 覆盖基线约 Acc .814、AP .737，更接近 source-view `c6_canon`，不对应摘要的 .826/.754。

Table 12 还混合了 LoRA-frozen 单种子扫描与 full fine-tuning 三种子 canonical。`model.py` 在 `enc_train=full` 时禁用 LoRA，因此 full-FT canonical 中的 LoRA rank 是无效参数；不能据此声称 rank 16 是 full-FT 的最优值。

## 四、基线可追溯性问题

1. 历史 `paper_results.jsonl` 中 BERT-CLS 与 BERT-NLI 的三个 seed 逐项完全相同。原代码在 NLI 骨干不可达时静默退回 `bert-base-chinese`，这很可能使两者成为同一个模型。新代码已改为正式运行时硬失败，并要求显式提供 `CLAIMARC_NLI_PATH`。
2. 历史 artifact 的 BERT-CLS AP 均值约 .7062，论文写 .682；ESIM artifact 均值约 .5229，论文写 .512。现有文件没有明确记录论文每一行到底从哪个结果版本汇编而来。
3. `paper_results.jsonl`、`paper_compiled.json`、`metrics_rich.json` 和 campaign6–8 属于不同代际，部分 canonical 为 .7168/.7189，部分为 .7363，论文则为 .7541。历史行普遍缺少 dataset hash、evidence policy、模型解析路径和代码提交号。
4. Qwen2.5-7B LoRA SFT 只有一条聚合结果/预测痕迹；当前 `src` 没有对应训练脚本、模板、优化配置、checkpoint 标识和三种子预测。
5. 网关 LLM 基线为单次调用；若论文保留“所有可学习系统三种子”的措辞，应明确 LLM 调用是否确定、缓存如何冻结以及重复性如何处理。
6. 当前复现快照没有 campaign6 跨域逐折 `.pt` bundle，只有聚合 JSON；无法从逐样本预测重新计算显著性和注入曲线。

## 五、已实施的代码修复

- CLAIMARC、微调基线、从零神经基线、冻结探针、LLM 和跨域 runner 均显式支持同一 `evidence_policy`。
- 规范化默认值固定为 `sources_only`；结果行写入数据集 SHA-256、标签/划分字段、room 分组、证据策略和解析后的模型标识。
- BERT-NLI 取消静默回退；若骨干缺失即中止正式运行。
- 训练预测 bundle 增加 provenance 块。
- campaign6–8 改用当前 Python、环境变量模型路径及 `results/fair_rerun` / `embeddings/fair_rerun`，不再写入数据目录或依赖 `/root/...`。
- argument-only 运行默认禁用，直到存在单独冻结、非空且通过审计的 arguments 数据集。

这些改动修复了未来运行的可比性，不会把旧结果自动变成公平结果；论文数值仍必须重跑后替换。
