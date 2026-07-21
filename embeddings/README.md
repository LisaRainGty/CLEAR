# 逐样本产物与嵌入库

本目录保存公平重跑导出的逐样本概率、检索嵌入和跨域 bundle（`*.pt`）。
所有审稿级产物统一位于 `fair_rerun/`，历史 bundle 已从发布树移出。

`fair_rerun/emb_geom/` 的每个 bundle 含 `train` / `val` / `test` 三划分的
`g`（检索嵌入）、`p`（前向概率）、`y`、`c`、`attr`、`pair_id` 和运行溯源字段。
其他子目录保存 Table 3 基线、消融、超参和跨域实验的对应预测 bundle。

重新生成：

```bash
python scripts/run_paper_suite.py --stages table3 ablation --execute
python scripts/run_paper_suite.py --stages analysis --execute
```

聚合和统计脚本会校验 pair-id 顺序、数据哈希和证据策略，不接受历史或不完整产物。
