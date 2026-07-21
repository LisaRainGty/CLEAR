# 复现操作说明

## 1. 校验快照

```bash
cd main
python3 scripts/unpack_data_archives.py --verify-only
python3 scripts/audit_reproducibility.py
```

报告写入 `results/audit/reproducibility_report.json`。当前三源协议预期状态为
`PASS`：数据规模、哈希、room 分组、`sources_only` 和 arguments 排除检查均应通过。

文件清单：

```bash
python3 scripts/build_manifest.py
sha256sum -c CHECKSUMS.sha256
```

第一条通过归档 SHA-256 和排序内容树 SHA-256 校验 raw/processed 的
18,130 个原始文件，不会占用额外解包空间。发布清单使用 `--hash-all`
为归档、final、论文、代码与结果逐文件计算哈希。

## 2. 环境

本次远程重跑使用 Linux、Python 3.9.16、PyTorch 2.0.0+cu117 与 NVIDIA A30
24 GB。训练依赖见 `requirements-train.txt`；`docs/archive/requirements_dev_snapshot.txt` 只是原开发环境归档，
不应用于这台 CUDA 11.7 主机。runner 会把实际 `pip freeze`、GPU、模型解析路径
每个独立队列写入不会互相覆盖的环境快照，例如
`results/fair_rerun/suite_environment_non_api.json` 与
`results/fair_rerun/suite_environment_llm.json`；相应失败清单同样按队列命名。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
cp env.example.sh env.sh
source env.sh
```

至少配置：

- `CLAIMARC_BGE_PATH`：BGE-large-zh-v1.5 本地路径或留空在线解析
- `CLAIMARC_BERT_PATH`
- `CLAIMARC_ROBERTA_PATH`
- `CLAIMARC_NLI_PATH`：真正的中文 NLI 预训练骨干；正式实验必需
- `MATPOOL_API_KEY`：仅 LLM/VLM 重放需要

## 3. 查看而不运行

```bash
python3 scripts/run_paper_suite.py
```

这会打印规范化主实验、基线和几何实验命令，不启动 GPU 训练。选择阶段示例：

```bash
python3 scripts/run_paper_suite.py --stages table3,ablation
```

## 4. 公平重跑

```bash
python3 scripts/run_paper_suite.py --execute
```

结果进入 `results/fair_rerun`，预测/嵌入进入 `embeddings/fair_rerun`。runner 为每个任务保存命令、UTC 时间、返回码和日志。所有论文任务固定 `sources_only`；历史 campaign6–8 入口和空 arguments 任务已从发布包移出。

统一入口包含 187 个默认无 API 任务：主模型、基线、几何、可靠性、22 个全三源 LoRA 设置、跨域聚合和论文 Table 5 的留主播检索库注入。可按阶段执行，例如：

```bash
python3 scripts/run_paper_suite.py --stages ablation,hparams --execute
python3 scripts/run_paper_suite.py --stages xdom --execute
```

LLM 网关未放入默认一键执行，因为会产生外部调用成本；可在确认 key、模型名和预算后显式运行：

```bash
python3 scripts/run_paper_suite.py --stages llm --execute
```

Qwen2.5-7B 行由 `src/models/qwen_sft.py` 以 4-bit QLoRA 和同一三源输入重跑。API 阶段另有 23 个任务；未配置密钥时应保持未运行，不用历史数值代替。

## 5. 结果验收

只有同时满足以下条件，数值才可写回论文：

1. 每行的 dataset SHA-256、evidence policy、split、seeds 和测试 pair_id 完全相同。
2. 模型/超参选择未读取测试指标。
3. 所有可学习模型三种子齐全；逐样本预测可重新计算所有表格指标。
4. BERT-NLI 的 resolved model 确为 NLI 骨干，与 BERT-CLS 不同。
5. Table 4–5 的每折/每 seed 原始 bundle 齐全；注入池与评估集无交集。
6. Figure 1、6、7 由同一个公平 canonical prediction bundle 生成。
7. 2,000 次配对 bootstrap 脚本、seed 与输出一并保存。

论文草稿目前只作为历史快照保留，不应在重跑前直接覆盖其数值。
