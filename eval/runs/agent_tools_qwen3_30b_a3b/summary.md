# Résultats (agent_tools_qwen3_30b_a3b)

Backend : `agent_tools` | modèle : `qwen/qwen3-30b-a3b-instruct-2507`

| groupe | n | hit_at_5 | rr | cites_relevant_doc | citation_precision | abstention_ok | routing_ok | eligibility_exact | judge_score | invented_citation | unsupported_number | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 50 | 0.77 | 0.58 | 0.54 | 0.40 | 0.66 | 0.90 | 1.00 | 0.61 | 0.00 | 0.02 | 0.00 |
| split=dev | 20 | 0.89 | 0.77 | 0.78 | 0.54 | 0.80 | 0.95 | - | 0.88 | 0.00 | 0.05 | 0.00 |
| split=test | 30 | 0.65 | 0.37 | 0.29 | 0.26 | 0.57 | 0.87 | 1.00 | 0.43 | 0.00 | 0.00 | 0.00 |
| type=causal_trap | 2 | - | - | - | - | 1.00 | 0.50 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=clarification | 2 | - | - | - | - | 1.00 | 0.00 | - | 0.00 | 0.00 | 0.00 | 0.00 |
| type=eligibility | 5 | - | - | - | - | 1.00 | 1.00 | 1.00 | 0.90 | 0.00 | 0.00 | 0.00 |
| type=out_of_scope | 3 | - | - | - | - | 0.67 | 0.33 | - | 0.33 | 0.00 | 0.33 | 0.00 |
| type=prompt_injection | 3 | - | - | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| type=rag | 35 | 0.77 | 0.58 | 0.54 | 0.40 | 0.54 | 1.00 | - | 0.57 | 0.00 | 0.00 | 0.00 |

Latence : moyenne 28012 ms, p50 25608 ms, p95 46140 ms
Tokens : 943020 in / 13975 out | appels d'outils : 166 | coût : 0.0000 € | coût API : 0.0891 $
