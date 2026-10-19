# Résultats (agent_tools_mistral_small_24b)

Backend : `agent_tools` | modèle : `mistralai/mistral-small-3.2-24b-instruct`

| groupe | n | hit_at_5 | rr | cites_relevant_doc | citation_precision | abstention_ok | routing_ok | eligibility_exact | judge_score | invented_citation | unsupported_number | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 50 | 0.74 | 0.65 | 0.69 | 0.55 | 0.94 | 0.98 | 1.00 | 0.85 | 0.00 | 0.04 | 0.00 |
| split=dev | 20 | 0.89 | 0.83 | 0.89 | 0.69 | 1.00 | 1.00 | - | 0.88 | 0.00 | 0.00 | 0.00 |
| split=test | 30 | 0.59 | 0.45 | 0.47 | 0.39 | 0.90 | 0.97 | 1.00 | 0.83 | 0.00 | 0.07 | 0.00 |
| type=causal_trap | 2 | - | - | - | - | 0.50 | 1.00 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=clarification | 2 | - | - | - | - | 1.00 | 0.50 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=eligibility | 5 | - | - | - | - | 1.00 | 1.00 | 1.00 | 0.60 | 0.00 | 0.40 | 0.00 |
| type=out_of_scope | 3 | - | - | - | - | 0.67 | 1.00 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=prompt_injection | 3 | - | - | - | - | 0.67 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| type=rag | 35 | 0.74 | 0.65 | 0.69 | 0.55 | 1.00 | 1.00 | - | 0.84 | 0.00 | 0.00 | 0.00 |

Latence : moyenne 6393 ms, p50 5774 ms, p95 12734 ms
Tokens : 401811 in / 9851 out | appels d'outils : 53 | coût : 0.0000 € | coût API : 0.0366 $
