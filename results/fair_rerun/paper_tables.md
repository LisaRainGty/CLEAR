# 论文全部实验表：三源证据公平重跑

> 本文件只汇总 `sources_only` (PARAM + OCR + VLM) 公平重跑。`PENDING` 表示对应 GPU/API 任务尚未成功完成，不会用历史异口径数字填补。

- Dataset SHA-256: `1eff4c58fff61ed85763f92ecd321f8d61a66026d32b97113cfce26f0c470f76`
- Rows: 4883
- Argument records: 0
- Accepted fresh RESULT rows: 77
- Rejected incomplete/off-protocol RESULT rows: 0

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

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| ESIM | 74.23±0.24 | 67.37±1.00 | 51.92±0.34 | 80.78±0.05 | 3 |
| Decomposable Attention | 75.79±1.50 | 60.01±3.09 | 59.61±3.40 | 82.52±0.87 | 3 |
| BERT-NLI | 79.42±0.91 | 67.83±3.24 | 67.77±1.35 | 87.71±0.28 | 3 |
| TextCNN | 77.70±0.64 | 61.25±2.24 | 61.35±0.65 | 84.60±0.52 | 3 |
| BiLSTM | 77.93±0.20 | 64.53±1.36 | 68.13±0.32 | 86.82±0.28 | 3 |
| BERT-CLS | 80.16±0.54 | 70.79±1.28 | 68.55±0.88 | 88.39±0.17 | 3 |
| RoBERTa-CLS | 80.74±0.40 | 70.86±2.07 | 68.94±1.87 | 88.73±0.06 | 3 |
| BGE frozen + LR | 79.42 | 68.00 | 67.34 | 87.01 | 1 |
| BGE frozen + SVM | 75.20 | 66.98 | 61.68 | 85.14 | 1 |
| BGE frozen + MLP | 76.84 | 66.78 | 63.50 | 86.07 | 1 |
| BGE frozen + kNN | 76.49 | 59.23 | 56.56 | 78.94 | 1 |
| Qwen-Flash zero-shot | 60.00 | 25.65 | 28.88 | 48.91 | 1 |
| Qwen-Flash five-shot | 57.31 | 32.78 | 28.80 | 49.90 | 1 |
| GPT-5.4 zero-shot | 58.25 | 29.31 | 29.68 | 48.96 | 1 |
| GPT-5.4 five-shot | 54.15 | 31.23 | 28.26 | 47.34 | 1 |
| Gemini-3.5-Flash zero-shot | PENDING | PENDING | PENDING | PENDING | PENDING |
| Gemini-3.5-Flash five-shot | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kimi-K2.6 zero-shot | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kimi-K2.6 five-shot | PENDING | PENDING | PENDING | PENDING | PENDING |
| Qwen2.5-7B QLoRA SFT | PENDING | PENDING | PENDING | PENDING | PENDING |
| CLAIMARC | 80.78±0.56 | 71.52±1.40 | 70.18±1.40 | 88.92±0.13 | 3 |

## Table 4a. Leave-one-category transfer

| System | Acc | F1pos | Macro-F1 | AUPRC | AUROC | folds |
|---|---:|---:|---:|---:|---:|---:|
| PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 4b. Leave-20-streamer transfer

| System | Acc | F1pos | Macro-F1 | AUPRC | AUROC | folds |
|---|---:|---:|---:|---:|---:|---:|
| PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

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

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| Canonical | 80.78±0.56 | 71.52±1.40 | 70.18±1.40 | 88.92±0.13 | 3 |
| w/o RACL | 81.21±1.00 | 71.58±1.94 | 70.93±1.70 | 89.19±0.46 | 3 |
| w/o reliability | 80.78±1.24 | 72.00±0.58 | 73.42±0.98 | 89.77±0.69 | 3 |
| w/o class balance | 80.98±0.22 | 70.80±1.66 | 70.04±0.95 | 88.89±0.07 | 3 |
| w/o four-tuple | 80.86±0.90 | 72.05±1.87 | 72.28±2.58 | 88.99±1.15 | 3 |
| BERT backbone | 80.31±0.57 | 71.89±1.48 | 69.53±0.38 | 88.58±0.38 | 3 |

## Table 8. Dual-stream ablations

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| Canonical | 80.78±0.56 | 71.52±1.40 | 70.18±1.40 | 88.92±0.13 | 3 |
| w/o fusion | 80.66±0.79 | 72.18±2.63 | 72.27±0.71 | 89.74±0.46 | 3 |
| Claim only | PENDING | PENDING | PENDING | PENDING | PENDING |
| Evidence only | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 9. RACL mining

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| Canonical | 80.78±0.56 | 71.52±1.40 | 70.18±1.40 | 88.92±0.13 | 3 |
| Hard positive | 81.17±0.35 | 72.63±0.60 | 69.12±0.71 | 87.90±1.13 | 3 |
| Same-attribute negative | 80.82±1.38 | 70.96±3.30 | 68.97±0.62 | 88.71±0.35 | 3 |
| Same-evidence-type negative | 81.44±0.11 | 72.28±0.62 | 69.46±1.37 | 88.69±0.40 | 3 |
| Kp=1 | 80.90±0.81 | 70.98±3.14 | 71.57±2.02 | 89.20±0.81 | 3 |
| Kp=5 | 81.05±0.78 | 71.34±1.53 | 68.40±1.13 | 88.64±0.57 | 3 |
| Kn=1 | 81.44±0.40 | 72.20±1.04 | 68.77±0.90 | 88.78±0.29 | 3 |
| Kn=10 | 81.09±0.28 | 70.86±1.63 | 70.55±0.87 | 89.06±0.23 | 3 |

## Table 10. Reliability counterfactuals

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| Canonical c | 80.78±0.56 | 71.52±1.40 | 70.18±1.40 | 88.92±0.13 | 3 |
| Uniform | 80.78±1.24 | 72.00±0.58 | 73.42±0.98 | 89.77±0.69 | 3 |
| Inverse | 82.14±0.40 | 72.84±1.90 | 73.38±3.99 | 90.08±0.95 | 3 |
| Permuted | PENDING | PENDING | PENDING | PENDING | PENDING |
| Binary | PENDING | PENDING | PENDING | PENDING | PENDING |
| Count only | PENDING | PENDING | PENDING | PENDING | PENDING |
| sqrt(c) | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 11

The current manuscript has no Table 11 (numbering gap).

## Table 12. LoRA-efficient hyperparameter sensitivity (all three-source)

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| Canonical LoRA (N2, h8, r16, lambda=.5, tau=.07, Kp3/Kn5, BCE) | PENDING | PENDING | PENDING | PENDING | PENDING |
| Fusion blocks N=1 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Fusion blocks N=3 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Fusion blocks N=4 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Attention heads=4 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Attention heads=16 | PENDING | PENDING | PENDING | PENDING | PENDING |
| LoRA rank=8 | PENDING | PENDING | PENDING | PENDING | PENDING |
| LoRA rank=32 | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda_CL=0.1 | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda_CL=0.3 | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda_CL=1.0 | PENDING | PENDING | PENDING | PENDING | PENDING |
| tau=0.05 | PENDING | PENDING | PENDING | PENDING | PENDING |
| tau=0.10 | PENDING | PENDING | PENDING | PENDING | PENDING |
| tau=0.20 | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kp/Kn=(1,1) | PENDING | PENDING | PENDING | PENDING | PENDING |
| Kp/Kn=(5,10) | PENDING | PENDING | PENDING | PENDING | PENDING |
| ASL | PENDING | PENDING | PENDING | PENDING | PENDING |
| Focal loss | PENDING | PENDING | PENDING | PENDING | PENDING |
| FFN GeLU | PENDING | PENDING | PENDING | PENDING | PENDING |
| cross-attention claim to evidence | PENDING | PENDING | PENDING | PENDING | PENDING |
| cross-attention evidence to claim | PENDING | PENDING | PENDING | PENDING | PENDING |
| Independent projections | PENDING | PENDING | PENDING | PENDING | PENDING |

## Table 13. Reliability-formula sensitivity (matched seed 0)

| 方法/设定 | acc | pos_f1 | auprc | auroc | n |
|---|---:|---:|---:|---:|---:|
| Canonical | 81.29 | 73.15 | 69.83 | 88.82 | 1 |
| k=1.5 | PENDING | PENDING | PENDING | PENDING | PENDING |
| k=6 | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda=0.1 | PENDING | PENDING | PENDING | PENDING | PENDING |
| lambda=0.6 | PENDING | PENDING | PENDING | PENDING | PENDING |
| rho=0.2 | PENDING | PENDING | PENDING | PENDING | PENDING |
| rho=0.6 | PENDING | PENDING | PENDING | PENDING | PENDING |
| phi=1.0 | PENDING | PENDING | PENDING | PENDING | PENDING |
| phi=1.5 | PENDING | PENDING | PENDING | PENDING | PENDING |

