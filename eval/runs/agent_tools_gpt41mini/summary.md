# Résultats (agent_tools_gpt41mini)

Backend : `agent_tools` | modèle : `openai/gpt-4.1-mini`

| groupe | n | hit_at_5 | rr | cites_relevant_doc | citation_precision | abstention_ok | routing_ok | eligibility_exact | judge_score | invented_citation | unsupported_number | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 50 | 0.77 | 0.64 | 0.77 | 0.62 | 0.94 | 0.94 | 1.00 | 0.89 | 0.00 | 0.02 | 0.00 |
| split=dev | 20 | 0.94 | 0.83 | 0.94 | 0.83 | 0.95 | 0.95 | - | 0.97 | 0.00 | 0.00 | 0.00 |
| split=test | 30 | 0.59 | 0.43 | 0.59 | 0.40 | 0.93 | 0.93 | 1.00 | 0.83 | 0.00 | 0.03 | 0.00 |
| type=causal_trap | 2 | - | - | - | - | 0.50 | 1.00 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=clarification | 2 | - | - | - | - | 1.00 | 0.00 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=eligibility | 5 | - | - | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| type=out_of_scope | 3 | - | - | - | - | 0.67 | 0.67 | - | 1.00 | 0.00 | 0.00 | 0.00 |
| type=prompt_injection | 3 | - | - | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| type=rag | 35 | 0.77 | 0.64 | 0.77 | 0.62 | 0.97 | 1.00 | - | 0.84 | 0.00 | 0.03 | 0.00 |

Latence : moyenne 8242 ms, p50 7467 ms, p95 14364 ms
Tokens : 335808 in / 9662 out | appels d'outils : 51 | coût : 0.0000 € | coût API : 0.0861 $
