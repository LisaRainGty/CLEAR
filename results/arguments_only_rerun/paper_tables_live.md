# 论文全部实验表：Arguments-only 公平重跑

> 本文件只汇总 `args_only` (supporting + refuting + evidence-gap arguments) 公平重跑。`PENDING` 表示对应 GPU/API 任务尚未成功完成，不会用历史异口径数字填补。

- Dataset SHA-256: `b6bc9a91f87da3a489af216a036e4d11bd02d7eb8895e9d7f2cd9d78e26bd618`
- Rows: 4883
- Argument records: 4883
- Accepted fresh RESULT rows: 42
- Rejected incomplete/off-protocol RESULT rows: 23

## Table 1. Dataset split and statistics

| Split | N | Positive | Positive rate | Rooms |
|---|---:|---:|---:|---:|
| train | 3636 | 879 | 24.17% | 92 |
| val | 392 | 122 | 31.12% | 5 |
| test | 855 | 249 | 29.12% | 11 |
| **All** | **4883** | **1250** | **25.60%** | **108** |

Evidence-source availability (0/1/2/3): 1483/1826/1118/456; reliability c mean/median/range: 0.456/0.480/[0.180, 0.820].
Construction sources: {'factrecords': 2278, 'factrecords_neg': 2605}; aligned-comment pairs: 2278 (46.65%).

## Table 2. Category distribution

| Category | N |
|---|---:|
| apparel_and_underwear | 1274 |
| baby_kids_and_pets | 715 |
| beauty_and_personal_care | 225 |
| digital_and_electronics | 270 |
| food_and_beverages | 426 |
| general | 827 |
| jewelry_and_collectibles | 84 |
| shoes_and_bags | 476 |
| smart_home | 324 |
| sports_and_outdoor | 262 |

## Table 3. In-domain main comparison

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESIM | 74.58±0.24 | 68.65±0.47 | 73.64±0.27 | 51.92±0.31 | 81.38±0.10 | 70.05±0.26 | 14.85±1.15 | 3 |
| Decomposable Attention | 76.26±1.35 | 60.74±4.62 | 71.80±2.08 | 63.66±0.81 | 85.37±0.62 | 69.27±1.76 | 13.78±1.93 | 3 |
| BERT-NLI | 79.26±1.17 | 69.34±1.79 | 76.81±1.03 | 68.90±1.17 | 87.89±0.47 | 74.16±1.10 | 15.08±1.58 | 3 |
| TextCNN | 77.50±1.31 | 65.27±2.53 | 74.32±1.67 | 64.65±0.85 | 84.98±0.60 | 71.68±1.87 | 10.78±3.58 | 3 |
| BiLSTM | 78.21±0.83 | 66.94±2.58 | 75.29±0.51 | 68.87±1.36 | 86.95±0.62 | 72.90±0.34 | 13.93±2.67 | 3 |
| BERT-CLS | 80.86±0.70 | 70.77±2.08 | 78.26±1.15 | 68.68±1.86 | 88.44±0.82 | 75.72±1.26 | 12.62±2.37 | 3 |
| RoBERTa-CLS | 80.70±0.42 | 70.62±1.80 | 78.11±0.72 | 72.87±0.40 | 89.10±0.19 | 75.53±0.86 | 12.52±0.63 | 3 |
| BGE frozen + LR | 78.25 | 71.12 | 76.83 | 68.60 | 87.60 | 74.12 | 14.81 | 1 |
| BGE frozen + SVM | 77.43 | 66.08 | 74.58 | 63.56 | 86.04 | 72.45 | 8.98 | 1 |
| BGE frozen + MLP | 79.06 | 69.61 | 76.82 | 69.57 | 87.69 | 74.99 | 10.94 | 1 |
| BGE frozen + kNN | 73.57 | 57.99 | 69.35 | 56.01 | 78.40 | 69.49 | 8.36 | 1 |
| Qwen-Flash zero-shot | 49.24 | 31.11 | 45.46 | 27.58 | 46.02 | 49.09 | 59.54 | 1 |
| Qwen-Flash five-shot | 57.66 | 26.12 | 48.23 | 28.92 | 48.38 | 47.03 | 55.12 | 1 |
| GPT-5.4 zero-shot | 61.64 | 27.11 | 50.54 | 30.49 | 50.61 | 48.56 | 54.67 | 1 |
| GPT-5.4 five-shot | 59.42 | 31.56 | 51.36 | 30.77 | 50.68 | 50.56 | 54.05 | 1 |
| Gemini-3.5-Flash zero-shot | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Gemini-3.5-Flash five-shot | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kimi-K2.6 zero-shot | 55.56 | 26.92 | 47.50 | 28.08 | 46.41 | 46.55 | 46.50 | 1 |
| Kimi-K2.6 five-shot | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Qwen2.5-7B QLoRA SFT (locked) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| CLAIMARC | 81.40±0.42 | 71.76±1.78 | 78.93±0.76 | 75.16±1.68 | 90.17±0.47 | 76.85±0.81 | 13.02±0.29 | 3 |

## Table 4a. Leave-one-category transfer

| System | Acc | F1pos | Macro-F1 | AUPRC | AUROC | folds |
|---|---:|---:|---:|---:|---:|---:|

## Table 4b. Leave-20-streamer transfer

| System | Acc | F1pos | Macro-F1 | AUPRC | AUROC | folds |
|---|---:|---:|---:|---:|---:|---:|

## Table 5. Gradient-free target-library injection

| Domain protocol | Condition | AP | AUC | F1 |
|---|---|---:|---:|---:|
| rooms | PENDING | PENDING | PENDING | PENDING |

## Table 6. Representation geometry

| Variant | Silhouette | Hard purity@10 | Alignment | Uniformity |
|---|---:|---:|---:|---:|
| w/o contrast | PENDING | PENDING | PENDING | PENDING |
| SupCon | PENDING | PENDING | PENDING | PENDING |
| RACL | PENDING | PENDING | PENDING | PENDING |

## Table 7. Core ablations

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical | 81.40±0.42 | 71.76±1.78 | 78.93±0.76 | 75.16±1.68 | 90.17±0.47 | 76.85±0.81 | 13.02±0.29 | 3 |
| w/o RACL | 82.11±0.36 | 72.98±1.04 | 79.79±0.28 | 74.62±1.23 | 90.32±0.22 | 77.70±0.39 | 13.91±1.21 | 3 |
| w/o reliability | 81.05±0.51 | 73.06±0.32 | 79.22±0.42 | 75.19±1.38 | 90.18±0.49 | 76.81±0.52 | 11.82±1.82 | 3 |
| w/o class balance | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| w/o four-tuple | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| BERT backbone | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 8. Claim/argument interaction ablations

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical | 81.40±0.42 | 71.76±1.78 | 78.93±0.76 | 75.16±1.68 | 90.17±0.47 | 76.85±0.81 | 13.02±0.29 | 3 |
| w/o fusion | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Claim only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Evidence only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Sources only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Sources + arguments | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 9. RACL mining

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical | 81.40±0.42 | 71.76±1.78 | 78.93±0.76 | 75.16±1.68 | 90.17±0.47 | 76.85±0.81 | 13.02±0.29 | 3 |
| Hard positive | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Same-attribute RACL retrieval | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Same-evidence-type negative | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kp=1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kp=5 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kn=1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kn=10 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 10. Reliability counterfactuals

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical c | 81.40±0.42 | 71.76±1.78 | 78.93±0.76 | 75.16±1.68 | 90.17±0.47 | 76.85±0.81 | 13.02±0.29 | 3 |
| Uniform | 81.05±0.51 | 73.06±0.32 | 79.22±0.42 | 75.19±1.38 | 90.18±0.49 | 76.81±0.52 | 11.82±1.82 | 3 |
| Inverse | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Permuted | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Binary | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Count only | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| sqrt(c) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 11

The current manuscript has no Table 11 (numbering gap).

## Table 12. LoRA-efficient hyperparameter sensitivity (args_only)

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical LoRA (N1, h8, r16, lambda=.5, tau=.07, Kp3/Kn5, BCE) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Fusion blocks N=2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Fusion blocks N=3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Fusion blocks N=4 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Attention heads=4 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Attention heads=16 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| LoRA rank=8 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| LoRA rank=32 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda_CL=0.05 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda_CL=0.20 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda_CL=0.50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| tau=0.05 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| tau=0.15 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| tau=0.20 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kp/Kn=(1,1) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kp/Kn=(5,10) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| ASL | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Focal loss | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| FFN GeLU | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| cross-attention claim to evidence | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| cross-attention evidence to claim | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Independent projections | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 13. Reliability-formula sensitivity (matched seed 0)

| 方法/设定 | acc | pos_f1 | macro_f1 | auprc | auroc | wF1 | ece | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical | 81.52 | 69.85 | 78.26 | 76.38 | 90.46 | 76.14 | 12.71 | 1 |
| k=1.5 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| k=6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda=0.1 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda=0.6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| rho=0.2 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| rho=0.6 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| phi=1.0 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| phi=1.5 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
