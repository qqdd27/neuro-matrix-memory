# LoCoMo evidence-hit@8 report

- date: 2026-09-14 14:09
- conversations: 10, questions scored: 607
- overall evidence-hit@8: **380/607 (62.6%)**

| Category | n | hit@8 |
|---|---|---|
| 1 single-hop | 85 | 44/85 (51.8%) |
| 2 temporal | 102 | 70/102 (68.6%) |
| 3 multi-hop | 22 | 7/22 (31.8%) |
| 4 open-domain | 252 | 166/252 (65.9%) |
| 5 adversarial | 146 | 93/146 (63.7%) |

## QA-accuracy (F1), n=607 sampled

| Category | n | avg F1 |
|---|---|---|
| 1 single-hop | 85 | 24.6% |
| 2 temporal | 102 | 5.5% |
| 3 multi-hop | 22 | 18.9% |
| 4 open-domain | 252 | 42.5% |
| 5 adversarial | 146 | 39.1% |

**Overall F1: 32.1%** (model=qwen2.5:1.5b)

## Date accuracy (calendar point), n=83

| Category | n | date-hit |
|---|---|---|
| 1 single-hop | 1 | 0/1 (0.0%) |
| 2 temporal | 82 | 52/82 (63.4%) |

**Overall date accuracy: 62.7%** (52/83)
