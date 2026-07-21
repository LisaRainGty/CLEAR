# CLAIMARC 论文复现包

本目录集中保存论文草稿、原始数据、全量中间产物、最终训练数据、代码、结果和公平重跑入口。原项目与参考项目未被改写；本目录是面向审稿复核的新快照。发布版只保留生成最终数据与论文结果所必需的产物；历史试验性 checkpoint 不作为论文结果。

## 当前结论

数据集已通过规模、标签、room 分组切分、哈希和全链重建核对。论文旧头条结果
**82.6 / 73.4 / 75.4 / 90.4** 不再作为公平主比较：它来自历史 `c6_args`
空证据视图，与其他模型的 PARAM/OCR/VLM 输入不一致。第一轮合法的
CLAIMARC 三种子结果为 Accuracy **80.78±0.56**、F1_pos **71.52±1.40**、
AP **70.18±1.40**、AUC **88.92±0.13**。

本复现包不使用也不生成 arguments，把 `sources_only` 固定为所有模型共享的输入：按 `[PARAM]` → `[OCR]` → `[VLM]` 固定顺序拼接每条样本实际存在的来源，不因缺少某源删除样本，不填充伪证据。所有最终表只接收本次公平队列产生的新结果。

完整证据与改进方案见：

- [实验与结果审计](docs/EXPERIMENT_AUDIT.md)
- [论文实验总表](docs/PAPER_EXPERIMENT_INVENTORY.md)
- [公平比较协议](docs/FAIR_COMPARISON_PROTOCOL.md)
- [数据来源与加工链](docs/DATA_PROVENANCE.md)
- [复现操作说明](docs/REPRODUCIBILITY.md)
- [参考项目结构采用说明](docs/REFERENCE_PROJECT_NOTES.md)
- [远程公平重跑记录](docs/REMOTE_RERUN.md)

## 目录

```text
main/
├── configs/        # 冻结的数据、模型与公平比较协议
├── data/
│   ├── archives/   # raw/processed 完整归档、哈希清单与解包说明
│   ├── index/      # 商品索引
│   └── final/      # 冻结评审输入、后续处理阶段与最终训练集
├── docs/           # 审计、来源、实验映射与复现说明
├── embeddings/     # 公平重跑的逐样本预测与嵌入
├── paper/          # 论文、图、LaTeX、结果日志与可读转换稿
├── results/        # 审计结果与公平重跑输出
├── scripts/        # 审计、清单、统一实验入口
└── src/            # 数据流水线、模型、基线与分析代码
```

## 最短复核路径

```bash
cd main
python3 scripts/unpack_data_archives.py --verify-only
PYTHONPATH=src python3 -m data_quality.rebuild_paper_dataset
python3 scripts/audit_reproducibility.py
python3 scripts/run_paper_suite.py
```

第一条不解包即流式校验 18,130 个 raw/processed 文件的内容树。
第二条在临时目录从 44 个冻结评审输入重建最终数据链并逐字节校验。
第三条生成 `results/audit/reproducibility_report.json`，预期为 `PASS`。第四条默认只打印完整 GPU
重跑命令；确认模型路径、CUDA 和环境后再加 `--execute`。

最终训练数据的固定 SHA-256：

```text
1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76
```

## 公平重跑状态

`scripts/run_paper_suite.py` 共冻结 187 个默认无 API 任务：审计 1、Table 3 主比较 28、消融 71、Table 12 LoRA 扫描 22、跨域 55、汇总/绘图 10。已删除不在论文中的品类注入附加统计和融合系数扫描，避免冗余产物。所有可学习主表模型运行 seeds 0/1/2，冻结探针按论文设计运行固定 seed 0。每个任务保存日志、命令、返回码、结果行与逐样本 bundle，之后自动生成全部论文表格和图。

商业/托管 LLM 共 23 个任务，由于当前远端没有 API key 且调用会产生费用，默认不运行。它们不会用历史异口径数字填充；未完成时在最终表中明确标为 `PENDING (API)`。

本地机器没有 NVIDIA GPU；公平重跑已转移到经过最长证据冒烟测试的
NVIDIA A30 远程环境。未成功产生状态文件和预测 bundle 的表格行仍标记为
`PENDING`，不会用历史异口径结果伪装完成。
