# Writing Style Notes Used for the Rewrite

This note records the section-level writing logic used to revise the CLAIMARC manuscript.

## Abstract

Reference style: FEVER (Thorne et al., 2018), SciFact (Wadden et al., 2020), and RGCL (Mei et al., 2024).

Observed habit: top NLP abstracts first name the task and missing benchmark/setting, then state the constructed resource or proposed model, then give one or two concrete empirical outcomes. They avoid broad motivation after the first sentence. The revised abstract therefore follows a four-step pattern: domain problem, task reframing, CLAIMARC mechanism, and headline numbers.

## Introduction

Reference style: SciFact for problem urgency and specialized-domain verification; FEVER for contrasting with adjacent tasks; marketing/IS papers (Aghakhani and Main; Wongkitrungrueng and Assarut) for construct grounding.

Observed habit: strong introductions move from a practical bottleneck to a conceptual mismatch, not from a list of techniques. They name the hidden assumption of prior work. The revised introduction therefore makes the central mismatch explicit: platform risk is tied to consumer-perceived deception rather than a mechanically verifiable contradiction. The three challenges are written as research obstacles rather than as implementation details.

## Related Work

Reference style: ACL/EMNLP papers use short thematic blocks that end with the paper's position; marketing papers foreground construct validity and boundary conditions.

Observed habit: each subsection should not merely summarize papers. It should close with what remains unsolved. The revised related work uses four blocks: livestream persuasive communication, fact verification, consumer reviews as measurement, and retrieval-contrastive learning. Each block explains why CLAIMARC needs a different supervision anchor or model geometry.

## Data and Task Formulation

Reference style: FEVER and SciFact introduce datasets by defining the unit, evidence, label, and annotation/reliability assumptions before giving counts.

Observed habit: empirical AI papers are clearest when the instance definition appears before the pipeline details. The revised version first defines a product-attribute pair, then describes claim, evidence, perceived-risk label, and reliability weight. Data construction is described only at the level needed to justify the task.

## Method

Reference style: ESIM-style claim-evidence papers for local comparison features; RGCL and supervised contrastive papers for retrieval-guided training; DPR for separating retrieval index from parametric scoring.

Observed habit: method sections should keep notation stable and present the objective as the natural answer to the challenge in the introduction. The revised method uses one notation system throughout: `X^c`, `X^e`, `h_c`, `h_e`, `g`, `c_i`. It states that the canonical model fully fine-tunes BGE and that RACL uses global same-label and opposite-label pools, not attribute partitions.

## Experiments

Reference style: RGCL and SciFact organize experiments by research questions and competing explanations.

Observed habit: main comparisons should be interpreted by model family, not by reading every row. The revised experiment section states four research questions, gives baselines as competing explanations, and reads the result table in groups: fact verification, single-stream classification, frozen retrieval probes, and LLMs.

## Conclusion

Reference style: top NLP conclusions are compact and return to the claims established by experiments rather than adding new motivation.

Observed habit: the conclusion should restate what was learned, what was contributed, and what remains limited. The revised conclusion maps back to RQ1-RQ4 and then separates problem-level, method-level, and deployment-level contributions.
