# 参考项目结构采用说明

本次规范化对照了 `reference/RGCL-main` 与 `reference/INFORMSJoC-2022.0285-sarcasm-cause`，采用的是可复现组织原则，不复制对方代码或数据。

从 RGCL 采用：清晰区分数据准备、训练实验与离线分析；把对比学习主实验和几何分析放在可执行 runner 中；保留预生成嵌入供后续分析复用。

从 INFORMS JoC 复现包采用：顶层区分 data/src/results/scripts；保留一个明确运行入口；把论文对象与脚本/结果逐项登记；强调数据快照、固定折/seed 和可复算输出。

CLAIMARC 的对应实现是：

- `configs/paper_fair.json`：唯一冻结协议；
- `configs/experiment_registry.json`：Table/Figure 到代码与结果的一一映射；
- `scripts/run_paper_suite.py`：统一 dry-run/执行入口；
- `scripts/audit_reproducibility.py`：数据与历史结果前置审计；
- `MANIFEST.tsv` 与 `CHECKSUMS.sha256`：归档清单；
- `results/fair_rerun` 与 `embeddings/fair_rerun`：新结果与历史结果隔离。
