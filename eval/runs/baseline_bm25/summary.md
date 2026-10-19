# Résultats (baseline_bm25)

Backend : `baseline` | modèle : `none (extractif)`

| groupe | n | hit_at_5 | rr | cites_relevant_doc | citation_precision | abstention_ok | routing_ok | eligibility_exact | judge_score | invented_citation | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 50 | 0.69 | 0.57 | 0.49 | 0.49 | 0.84 | 0.86 | 0.00 | - | 0.00 | 0.00 |
| split=dev | 20 | 1.00 | 0.83 | 0.72 | 0.72 | 0.90 | 1.00 | - | - | 0.00 | 0.00 |
| split=test | 30 | 0.35 | 0.28 | 0.24 | 0.24 | 0.80 | 0.77 | 0.00 | - | 0.00 | 0.00 |
| type=causal_trap | 2 | - | - | - | - | 0.50 | 0.50 | - | - | 0.00 | 0.00 |
| type=clarification | 2 | - | - | - | - | 0.00 | 1.00 | - | - | 0.00 | 0.00 |
| type=eligibility | 5 | - | - | - | - | 1.00 | 0.00 | 0.00 | - | 0.00 | 0.00 |
| type=out_of_scope | 3 | - | - | - | - | 0.00 | 1.00 | - | - | 0.00 | 0.00 |
| type=prompt_injection | 3 | - | - | - | - | 0.33 | 0.67 | 0.00 | - | 0.00 | 0.00 |
| type=rag | 35 | 0.69 | 0.57 | 0.49 | 0.49 | 1.00 | 1.00 | - | - | 0.00 | 0.00 |

Latence : moyenne 38 ms, p50 20 ms, p95 40 ms
Tokens : 0 in / 0 out | appels d'outils : 50 | coût : 0.0000 €
