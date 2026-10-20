# Avancement — Projet Assistant HLM RAG & Agent

> Ce fichier est le **journal opérationnel** du projet. Il doit être mis à jour après chaque tâche significative. `CONTEXT.md` contient les règles et décisions durables ; `AVANCEMENT.md` contient l'état réel, les résultats, les problèmes rencontrés et les prochaines actions.

## 1. État global

**État de référence au 21 octobre 2026 — projet livré et validé sans notebook :** le corpus final contient **455 documents** (117 Service-Public et 338 LEGI) et **1 357 chunks** selon `fixed_300_60`. Le retrieval `search_docs` associe BM25, embeddings E5, fusion RRF et reranking MiniLM. Les backends RAG et agent utilisent `qwen3:4b-instruct` via Ollama, des sorties structurées Pydantic, des citations contrôlées et une stratégie d'abstention. Les calculs d'éligibilité, les statistiques RPLS et les prix DVF sont produits par des outils Python déterministes.

**Interfaces et bonus :** l'application `app/streamlit_app.py` affiche les réponses, les sources et les traces d'outils. Le serveur `src/mcp_server.py` expose **quatre outils MCP** (recherche, éligibilité, RPLS, DVF), validés dans MCP Inspector v2.11.0. La connexion à Claude Desktop reste une perspective non testée.

**Validation (21 octobre, branche sans notebook validée) :** **215 tests réussis, 2 désélectionnés** ; les 9 tests MCP réussissent également. La version antérieure livrée le 20 octobre comptait 195 tests réussis et un désélectionné. ; `eval/check_golden.py` valide la **structure de 52 questions** (20 `dev`, 32 `test`). Cette validation ne garantit pas la justesse de toutes les réponses. Les limites juridiques observées pour `q021` et `q022` restent documentées. Le run local `agent` de 52 questions ne vaut pas évaluation exhaustive de `agent_tools --provider ollama`.

**Livraison :** PR #4 et #5 fusionnées ; bonus MCP intégré au commit `32ec0a3`. Le RAG amélioré par Alexis et les tests Anthropic sont intégrés au code de la version sans notebook ; les tests API Anthropic réels restent à effectuer. Un commit d'évaluation supplémentaire (`24702f`) est la **cible actuelle du tag `v1.0`** publié le 20 octobre 2026.

**Soutenance :** une **vidéo de secours de Streamlit a été enregistrée** ; une **démonstration Streamlit en direct est prévue le 21 octobre 2026**. La vidéo servira de repli en cas de problème technique ou de latence.

**Travaux ouverts ou facultatifs :** évaluation exhaustive de `agent_tools` en local, comparaison DINUM, analyse d'une variante de chunking structurelle et connexion éventuelle à un client MCP conversationnel externe. Un contrôle automatisé de sécurité Git a été réalisé.

**Échéances :** gel du code le 20 octobre 2026 à 23 h 59 (tag `v1.0` créé) ; soutenance le 21 octobre 2026.

## 2. Tableau de progression

### Cadrage

- [x] Énoncé du projet analysé.
- [x] Sujet spécialisé choisi : **Assistant HLM / logement social**.
- [x] Architecture générale définie : RAG documentaire + outils déterministes + agent.
- [x] Sources principales identifiées : Service-Public, LEGI, RPLS, DVF, plafonds HLM.
- [x] Limites principales identifiées : absence de disponibilité HLM temps réel, DVF ≠ loyers, barèmes annuels, prudence causale.

### Environnement

- [x] Python retenu comme langage principal.
- [x] Google Colab retenu pour le prototypage.
- [x] Google Drive utilisé pour la persistance des données importantes.
- [x] Dépôt GitHub de référence connecté : `Charles-Scannavino/Projet_Generative_AI`.
- [x] Arborescence GitHub structurée en `Docs/`, `notebooks/`, `eval/`, `data/`, `src/` et `app/`.
- [x] Décision prise de modulariser ensuite le notebook en fichiers `.py`.
- [x] Préparer l'environnement local final avec `uv` (`pyproject.toml`, `uv.lock`, Python 3.12.7 vérifié).
- [x] Vérifier l'exécution locale du projet sur laptop (**MacBook Pro M1, 16 Go de mémoire unifiée** ; PyTorch **MPS** opérationnel).
- [x] Vérifier la présence de **Homebrew (`brew`)** sur la machine locale.
- [x] Mettre en place Ollama / modèle local (`qwen3:4b`, 2,5 Go, MacBook Pro M1 16 Go).
- [x] Exécuter un premier test interactif avec `ollama run qwen3:4b`.
- [x] Tester l'API locale Ollama depuis Python dans le projet (appels réels via les backends RAG et agent).

### Service-Public — corpus brut

- [x] Jeu de données officiel Service-Public Particuliers identifié.
- [x] Fichier `vosdroits-latest.zip` téléchargé.
- [x] Premier essai d'extraction vers Google Drive.
- [x] Problème de performance identifié : milliers de petits fichiers écrits sur Drive.
- [x] Solution retenue : extraire temporairement dans `/content/` sur Colab.
- [x] Parsing initial des métadonnées XML réalisé.
- [x] Création d'un DataFrame Service-Public : **3 004 fiches × 9 colonnes**, puis 10 colonnes après ajout de `search_text`.
- [x] Premier filtre HLM strict exécuté.
- [x] Résultat observé : **24 fiches HLM candidates**.
- [x] Quantification du thème `Logement` : **350 fiches** avec le filtre `subject` contenant `Logement`.
- [x] Test d'un périmètre logement élargi par mots-clés : **505 fiches candidates**.
- [x] Contrôle visuel initial : le filtre élargi contient des faux positifs hors cœur logement/HLM.
- [x] Décision provisoire : utiliser les **350 fiches dont `subject` contient `Logement`** comme corpus Service-Public de travail.
- [x] Métadonnées de ce corpus sauvegardées dans `data/processed/service_public_logement_metadata.csv`.
- [x] Inspecter un échantillon reproductible de 20 documents ainsi que les extrêmes de longueur ; constat : le thème `Logement` contient de nombreuses fiches hors cœur HLM.
- [x] Extraire le texte complet des 350 fiches retenues (première version du parser).
- [x] Nettoyer et normaliser les textes (normalisation de base + correction ciblée des redondances structurelles parent/enfant).
- [x] Contrôler les documents vides, anormalement courts ou anormalement longs (0 vide ; extrêmes inspectés).
- [x] Sauvegarder le corpus textuel Service-Public final dans `data/processed/` (`service_public_hlm_documents.jsonl`, **117 documents**).

### LEGI

- [x] Identifier précisément le sous-ensemble du Code de la construction et de l'habitation utile au projet.
- [x] Télécharger / récupérer les articles (Parquet CCH AgentPublic/LEGI).
- [x] Définir l'unité documentaire : **1 `doc_id` = 1 article juridique reconstruit**.
- [x] Parser les métadonnées et reconstruire le texte des articles multi-chunks.
- [x] Générer le corpus LEGI HLM final : `data/processed/legi_hlm_documents.jsonl` (**338 articles**).
- [x] Vérifier que l'ensemble Service-Public + LEGI est cohérent avec la cible d'environ 500 documents : **455 documents**.

### Chunking documentaire

- [x] Lire manuellement un échantillon d'au moins 20 documents.
- [x] Définir plusieurs stratégies de chunking.
- [x] Comparer les trois stratégies de chunking `fixed_300_60`, `paragraph_400_80` et `fixed_500_100` (résultats dans `eval/bm25_chunking/`).
- [ ] Évaluer séparément une variante hiérarchique propre à la structure Service-Public / LEGI (non réalisée dans `v1.0`).
- [x] Conserver les métadonnées dans chaque chunk (`chunk_id`, `document_id`, source, titre, article, URL, position, texte et `retrieval_text`).
- [x] Comparer les stratégies sur le golden set avec BM25.
- [x] Figer le chunking final : `fixed_300_60` (300 mots, overlap 60).


### Golden set

- [x] Brouillon initial de 5 questions créé.
- [x] Golden set actif enrichi à **20 questions** dans `eval/golden.jsonl`.
- [x] Répartition actuelle : **18 RAG**, **1 hors périmètre**, **1 prompt injection**.
- [x] Les questions RAG sont ancrées sur des identifiants Service-Public réels.
- [x] Les cas d'éligibilité, RPLS et DVF non encore vérifiables sont conservés dans `eval/golden_backlog_pending.jsonl`.
- [x] Golden set v2 (Gaby) retenu le 17 octobre : **52 questions**, splits `dev` / `test`, champs `expected_tool`, `should_abstain`, `expected` (la v1 enrichie d'Alexis, 33 questions, est abandonnée).
- [x] Porter le golden set à **52 questions** (20 `dev`, 32 `test`) et vérifier sa structure avec `eval/check_golden.py`.
- [x] Couvrir :
  - factuel RAG ;
  - éligibilité ;
  - RPLS ;
  - DVF ;
  - absence de réponse ;
  - prompt injection ;
  - pièges causaux ;
  - abstention.
- [x] Vérifier les `gold_docs` avec les identifiants réels du corpus (**15/15 présents**).
- **Règle permanente :** ne pas inventer de réponses ou de sources manquantes avant validation des données.

### Retrieval

- [x] Implémenter BM25.
- [x] Mesurer BM25 sur le golden set.
- [x] Implémenter le dense retrieval.
- [x] Choisir et documenter le modèle d'embedding : `intfloat/multilingual-e5-small` (**384 dimensions**), exécuté via `sentence-transformers` sur **MPS**.
- [x] Comparer dense et BM25.
- [x] Implémenter la fusion hybride (`src/retrieval/hybrid.py`).
- [x] Tester une méthode de fusion, en priorité **RRF (Reciprocal Rank Fusion)** : k = 60, 100 candidats par retriever.
- [x] Ajouter un reranker (`src/retrieval/reranker.py`) : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` retenu, `BAAI/bge-reranker-v2-m3` comparé.
- [x] Comparer (`eval/eval_retrieval.py`, résultats dans `eval/retrieval/`) :
  - BM25 ;
  - dense ;
  - hybride ;
  - hybride + reranker.
- [x] Mesurer `Recall@5`, `Recall@10` et `MRR` pour BM25 et dense.
- [ ] Comparer avec la baseline DINUM si possible.

### RAG / génération

- [x] Construire `search_docs` (`src/retrieval/search.py`, contrat Pydantic dans `src/retrieval/schemas.py`, 22 tests dans `tests/test_retrieval.py`).
- [x] Installer et valider le modèle génératif local via Ollama : **`qwen3:4b`**.
- [x] Intégrer Ollama au RAG modulaire (`src/agent/rag.py`, modèle de démonstration `qwen3:4b-instruct`).
- [x] Générer des réponses à partir des passages retrouvés et limiter les affirmations non étayées (des erreurs de qualité résiduelles sont documentées).
- [x] Ajouter et contrôler les citations documentaires.
- [x] Mettre en place une stratégie d'abstention lorsque les données disponibles sont insuffisantes.
- [x] Valider les sorties structurées avec Pydantic.

### Outils déterministes

- [x] Télécharger les plafonds PLAI / PLUS / PLS : barème 2026 saisi dans `data/processed/plafonds_hlm_2026.csv` (ANIL pour PLAI/PLUS, Service-Public F869 pour PLS).
- [x] Implémenter `check_eligibility` (`src/tools/eligibility.py`).
- [x] Écrire des tests unitaires sur plusieurs configurations de foyer (`tests/test_tools.py`).
- [x] Télécharger / préparer RPLS (`src/ingestion/rpls.py`, `src/preprocessing/build_rpls_stats.py` → `data/processed/rpls_communes.csv`).
- [x] Implémenter `social_housing_stats` (`src/tools/social_housing.py`).
- [x] Télécharger / préparer DVF (`src/ingestion/dvf.py`, `src/preprocessing/build_dvf_stats.py` → `data/processed/dvf_communes.csv`).
- [x] Implémenter `market_prices` (`src/tools/market_prices.py`).
- [x] Toujours renvoyer l'année et la source des données (`provenance` + `caveats` dans chaque sortie).
- [x] Valider strictement les entrées des outils contre la prompt injection (`tests/test_tool_security.py`).

### Agent

- [x] Définir les descriptions des outils (`prompts/tool_descriptions.yaml`).
- [x] Implémenter le routage de l'agent (tool calling, `src/agent/agent.py`, backends OpenRouter et Ollama).
- [x] Vérifier que :
  - question documentaire → `search_docs` ;
  - éligibilité → `check_eligibility` ;
  - statistiques HLM → `social_housing_stats` ;
  - marché immobilier → `market_prices`.
  (routage exact 0,91, rappel 1,00 sur 33 questions avec `openai/gpt-4.1-mini`)
- [x] Tracer les appels d'outils (`AssistantResponse.tool_calls`, contrat commun).
- [x] Mesurer l'exactitude du routage (`routing_ok` dans `eval/run_eval.py`, backend `agent_tools`).
- [x] Garde-fous contre la prompt injection (parsing strict, registre fermé, quotas, sorties assainies, citations validées, ancrage des chiffres) : `tests/test_agent.py`, `tests/test_tool_security.py`.
- [x] Valider `agent_tools` avec Ollama / `qwen3:4b-instruct` (18 octobre, MacBook Air M2 8 Go) : API native `/api/chat`, contexte 8 192, repli par sortie structurée.
- [ ] Réaliser une évaluation exhaustive de **`agent_tools --provider ollama`** comparable aux runs API (un run `agent` local sur 52 questions existe, mais ne répond pas à ce besoin précis).

### Évaluation complète

- [x] Créer `eval/run_eval.py` et documenter son utilisation.
- [x] Prendre en charge la commande demandée :
  ```bash
  uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
  ```
- [x] Permettre la génération de `eval/results.jsonl` dans le dossier de sortie (les résultats de référence sont aussi stockés dans `eval/runs/`).
- [x] Enregistrer par question :
  - answer ;
  - cited sources ;
  - retrieved document IDs ;
  - tokens in/out ;
  - latency ;
  - tool calls ;
  - stop reason ;
  - coût.
- [x] Évaluer la justesse sur les runs API avec un juge LLM (sans étendre leurs scores aux autres configurations).
- [x] Étudier au moins cinq catégories de cas problématiques (retrieval, génération, abstention, exactitude numérique et causalité).
- [x] Comparer qualité, coût et latence sur les configurations réellement évaluées.

### Interface et livraison

### Bonus évoqués par l'enseignant

En complément du périmètre principal, deux pistes peuvent apporter une valeur supplémentaire au projet : **exposer la solution via un serveur MCP**, afin de pouvoir brancher notre assistant et ses outils à différents LLM compatibles, et **préparer une mise en production simple**, de façon à ce qu'un utilisateur puisse lancer et utiliser la solution finale depuis son propre ordinateur sans dépendre de notre environnement de développement. Ces éléments sont considérés comme des bonus et ne doivent pas retarder les fonctionnalités principales du RAG, de l'agent et de l'évaluation.


- [x] Créer l'interface Streamlit (`app/streamlit_app.py`) et la valider sur des scénarios réels.
- [x] Afficher réponse, sources et trace des appels de l'agent dans Streamlit.
- [x] README initial du dépôt rédigé et mis à jour avec l'architecture, l'état du projet et le workflow.
- [x] Documenter les noms et les principales configurations des modèles retenus dans `README.md`.
- [x] Ajouter `.env.example` (fichier présent dans le dépôt).
- [x] `.gitignore` adapté au projet (Python/Colab, données extraites, index, résultats d'évaluation, secrets).
- [x] Effectuer un contrôle automatisé de sécurité des fichiers suivis et de l'historique Git : `.env` ignoré, vérification des fichiers sensibles versionnés et scan Gitleaks exécuté sur 128 commits. Ce contrôle ne garantit pas l'absence absolue de secrets.
- [x] Préparer et enregistrer une vidéo de secours Streamlit ; démonstration en direct prévue à l'oral du 21 octobre (confirmé par le présentateur).
- [x] Créer et publier le tag `v1.0` (cible vérifiée : commit `24702f` ; le commit `32ec0a3` correspond à l'intégration du bonus MCP).

## 3. Résultats observés à ce jour

### Vue globale du corpus Service-Public

```text
Fichiers XML extraits                         : 5 574
Fiches Service-Public F*.xml parsées          : 3 004
Fiches du thème Logement                      : 350
Candidats logement élargi par mots-clés       : 505
Candidats HLM strict                          : 24
Corpus Service-Public de travail              : 350
```

Le notebook contient actuellement **41 cellules**. Les traitements observés vont jusqu'à la sauvegarde des métadonnées des 350 fiches logement dans :

```text
data/processed/service_public_logement_metadata.csv
```

### Sélection HLM stricte

Le premier filtre par mots-clés sur les métadonnées Service-Public a retourné :

```text
Nombre de fiches HLM candidates : 24
```

Exemples visibles dans le résultat :
- `F12006` — Aide personnalisée au logement (APL) ;
- `F1280` — Allocation de logement sociale (ALS) ;
- `F37237` — Dalo ;
- `F18005` — Droit au logement opposable (Dalo) : ...

**Interprétation :**
- le filtre fonctionne ;
- il est trop strict pour constituer à lui seul le corpus final ;
- la prochaine étape est d'analyser toutes les fiches relevant du sujet `Logement` et de définir un périmètre élargi cohérent ;
- LEGI pourra compléter le corpus documentaire.

## 4. Problèmes rencontrés et solutions

### Extraction du ZIP trop lente

**Symptôme :**
`zip_ref.extractall(EXTRACT_PATH)` prenait énormément de temps lorsque `EXTRACT_PATH` était situé dans Google Drive.

**Cause probable :**
Google Drive est lent pour l'écriture de milliers de petits fichiers.

**Solution retenue :**
extraire le ZIP dans le stockage local Colab :

```python
EXTRACT_PATH = "/content/service_public_xml"
```

Puis traiter localement les XML et ne sauvegarder sur Drive que les résultats utiles et consolidés.

**Statut :** ✅ résolu.

### Test du modèle local sans RAG

**Symptôme :** le premier test de `qwen3:4b` sur « Explique en français ce qu'est le DALO » n'a pas identifié correctement le **Droit au logement opposable** et a proposé des interprétations erronées.

**Interprétation :** ce résultat constitue une baseline utile : le LLM local seul ne doit pas être utilisé comme source de vérité sur le domaine HLM.

**Solution retenue :** utiliser `qwen3:4b` comme composant de génération/orchestration, mais lui fournir les connaissances métier via le **RAG Service-Public + LEGI**, avec citations et abstention lorsque les sources sont insuffisantes.

**Statut :** ✅ Ollama et le modèle sont opérationnels ; intégration au pipeline RAG encore à faire.

## 5. Prochaine action immédiate

Le retrieval est terminé : `search_docs` (BM25 + dense + RRF + reranker) est implémenté, évalué, testé et documenté.

Ablation finale sur les **18 questions RAG** du golden set (chunking `fixed_300_60`, métriques au niveau document, latence par requête sur MacBook Pro M1 / MPS) :

```text
Méthode                 R@1     R@5     R@10    MRR     p95/requête
BM25                    0.6944  1.0000  1.0000  0.8306      3 ms
Dense E5                0.5833  0.9444  1.0000  0.7394     13 ms
Hybride RRF             0.7500  0.9444  1.0000  0.8524     11 ms
Hybride + MiniLM        0.8611  1.0000  1.0000  0.9444    503 ms   <- configuration par défaut
Hybride + bge           0.9167  1.0000  1.0000  0.9722   4428 ms
```

Prochaines étapes :

1. tester l'API locale Ollama depuis Python ;
2. brancher `search_docs` sur `qwen3:4b` : réponse fondée uniquement sur les passages récupérés, citations (`document_id`, `url`), abstention si les passages sont insuffisants ;
3. créer `eval/run_eval.py` en enregistrant `SearchResponse.document_ids` comme documents récupérés ;
4. vérifier sur q020 que les passages restent traités comme des données et non comme des instructions.

Le code dense et ses résultats sont versionnés. Depuis le nettoyage d'historique du 13 octobre, le dépôt principal et le dépôt `review` conservent volontairement **des historiques Git différents** : le dépôt principal utilise l'historique nettoyé, tandis que `review` conserve l'historique d'origine. L'objectif est de maintenir **le même contenu fonctionnel** dans les deux dépôts sans force-push de l'un vers l'autre.

## 6. Règle de mise à jour de ce fichier

Mettre à jour `AVANCEMENT.md` après chaque étape significative.

Chaque mise à jour doit préciser :
- ce qui a été fait ;
- le résultat obtenu ;
- les éventuels chiffres ou métriques ;
- les problèmes rencontrés ;
- la solution adoptée ;
- la prochaine action.

Ne jamais remplacer un résultat réel par une hypothèse. Une tâche reste `[ ]` tant qu'elle n'a pas été effectivement exécutée ou confirmée.

## 7. Journal chronologique

### 18 octobre 2026 — runs OpenRouter jugés, compatibilité Ollama (branche `alexis`)

- **Juge via OpenRouter** : `run_eval.py --judge --judge-provider openrouter --judge-model google/gemini-2.5-flash` (sortie JSON stricte, coût du juge enregistré). Le juge est d'une autre famille que les modèles évalués, pour limiter le biais d'auto-préférence (Zheng et al., 2023, arXiv:2306.05685). Environ 0,014 $ par run de 50 questions.
- **Runs `agent_tools`** (50 questions, `eval/runs/`) :
  - `openai/gpt-4.1-mini` : justesse **0,89** (42 correctes, 5 partielles, 3 incorrectes : q024, q029, q047), Hit@5 0,94 / 0,59, abstention 0,94, éligibilité exacte 1,00, 0,086 $ ;
  - `mistralai/mistral-small-3.2-24b-instruct` : justesse 0,85, routage 0,98, 0,037 $ ; **plafonds inventés** dans 2 réponses d'éligibilité sur 5 (q031 : « éligible au PLUS » à 30 000 € à Paris, faux) — signalés par `unsupported_number` mais réponse fausse ;
  - `qwen/qwen3-30b-a3b-instruct-2507` : justesse 0,61, abstention 0,66, 28 s par question ; tool calling erratique (46 appels répétés, 32 communes inventées refusées, 12 questions RAG en limite d'étapes).
- **Incident** : le Mac d'Alexis (**MacBook Air M2, 8 Go**, et non le MacBook Pro 16 Go de référence) a subi un *kernel panic* (watchdog, swap saturé : disque à 98 %) pendant qu'Ollama, un run d'évaluation et un test tournaient en parallèle. Règle retenue : un seul processus lourd à la fois sur une machine à 8 Go. Le run Mistral interrompu a été relancé.
- **Compatibilité Ollama** (Ollama 0.40.6, `qwen3:4b-instruct`) :
  1. *Contexte* : l'endpoint compatible OpenAI `/v1` ignore `num_ctx` et applique le contexte du serveur (4 096 tokens par défaut ici) ; le 2e tour de l'agent (5 205 tokens) était refusé (erreur 400). Le fournisseur `ollama` passe désormais par l'API native `/api/chat` (client de Gaby, `OLLAMA_NUM_CTX` = 8 192 par requête).
  2. *Réponse finale* : le modèle 4B répond en texte libre au lieu d'appeler `final_answer` (citations perdues). Repli : reformulation en sortie JSON contrainte par le schéma de la réponse finale, puis validation habituelle.
  3. *Erreurs* : les erreurs Ollama sont gérées comme les erreurs d'API ; pas de contrôle d'ancrage des chiffres sur un message d'erreur.
  - Vérification sur 8 questions : `agent_tools` (Dalo cité F18005, éligibilité, DVF Villeurbanne 3 567 €/m², demande de précision, refus d'injection), `agent` (RPLS Créteil via `social_housing_stats`, éligibilité) et `rag` corrects ; 9 à 75 s par question. Reste : le refus d'injection n'est pas marqué comme abstention.
- **Tests** : 182 tests hors intégration (+ adaptateur Ollama, repli structuré, juge) et 1 test d'intégration OpenRouter.

### 17 octobre 2026 — intégration de `main` et de `gaby-ui` dans la branche `alexis`

- **Intégration** : `main` (RAG validé, PR #1) puis `gaby-ui` (golden v2, `run_eval.py`, contrat `AssistantResponse`, agent local, UI Streamlit) intégrés en avance rapide ; travail d'Alexis réappliqué par-dessus, non commité. Golden set : **v2 de Gaby conservée telle quelle** (les 13 questions outils de la v1 d'Alexis sont abandonnées).
- **Contrat unique `AssistantResponse`** : l'agent à outils d'Alexis devient le backend **`agent_tools`** de `ask()` (`src/agent/tool_agent.py`) ; `AgentResult` et `eval/eval_agent.py` sont retirés. Ajout rétrocompatible de deux champs optionnels : `cost_usd` (coût réel OpenRouter) et `prompt_version`. `run_eval.py` : option `--provider`, `cost_usd` enregistré et totalisé. UI : choix du fournisseur pour `agent_tools`, coût API affiché.
- **`check_eligibility` unifié** (implémentation d'Alexis, conventions de Gaby) : zones et financements en minuscules, sortie `categorie` / `plafonds` / `eligible` / `verdict` ; entrée `HouseholdSituation` (en couple, âges, personnes à charge) pour tous les agents ; `EligibilityError` = `ToolError` (sous-classe de `ValueError`). Les 11 tests de Gaby passent sans modification.
- **Comparaison des deux implémentations** (170 cas) : 104 divergences, dont des erreurs de zone de l'ancienne version (zone limitrophe décidée sur le seul nom) — Saint-Denis (11) et « Montreuil, 62 » classés « Paris et communes limitrophes » ; Quimper, Saint-Malo et « Saint-Ouen-sur-Seine » refusés (table de 23 villes). Divergence d'interprétation : personne seule + 1 personne à charge + handicap = catégorie 3 (lecture littérale des définitions, retenue) contre 4 ; à confirmer sur Légifrance. Écart de 1 € sur le PLS au-delà de 6 personnes (ordre arrondi / supplément). Tests de non-régression ajoutés.
- **Outils RPLS / DVF branchés sur l'agent local de Gaby** (`stats_tools()` pointe vers `social_housing_stats` et `market_prices`).
- **Défauts trouvés à l'évaluation `agent_tools` et corrigés** :
  1. *Calcul confié au LLM* : `jeune_menage=true` pour un couple de 40 + 38 ans (q033 : catégorie 3 au lieu de 2, PLS annoncé à tort). Le LLM fournit désormais les âges, Python calcule.
  2. *Filtre de source* : 50 recherches sur 53 restreintes à Service-Public par le modèle (Hit@5 LEGI = 0,14). `search_docs` n'expose plus que la requête.
  3. *Commune inventée* : « Paris » ajouté à « Je gagne 25 000 €, ai-je droit à un HLM ? » (q035, q052) dès que la commune est devenue obligatoire. Une commune absente de la question est désormais refusée par l'exécuteur.
  4. *Messages d'erreur* : les erreurs de nos validateurs sont renvoyées en clair au modèle (au lieu de « value_error »).
- **Résultats `run_eval.py`** (50 questions, `openai/gpt-4.1-mini`, run `eval/runs/agent_tools_gpt41mini/`) : Hit@5 **1,00** (dev) / **0,59** (test, baseline BM25 0,35) ; MRR 0,89 / 0,43 ; abstention correcte **0,94** ; routage **0,94** ; éligibilité exacte **1,00** (5/5) ; citations inventées **0** ; chiffres non ancrés **0** ; erreurs **0** ; latence p50 4,1 s, p95 4,9 s ; 340 712 tokens in / 10 037 out ; **0,066 $**.
- **Erreurs restantes** : q019 (disponibilité à Créteil), q039 (montant exact de l'APL) et q042 (causalité HLM / délinquance) : la réponse dit bien que les données ne permettent pas de conclure, mais `abstained=false` ; q035 et q052 : demande de précision correcte, mais `routing_ok=0` car la métrique compte l'appel tenté puis refusé par le garde-fou. RAG test : Hit@5 0,59, à améliorer côté retrieval (F869, articles LEGI).
- **Branche `charles`** (non intégrée) : un `src/agent/rag.py` différent de celui de Gaby (`RAGResponse`), à réconcilier par l'équipe.
- **Tests** : 178 tests hors intégration (dont 17 de Gaby) ; ruff sans nouvelle alerte sur les fichiers de Gaby modifiés.

### 13 octobre 2026 — agent (branche `alexis`)

- **Architecture** : boucle de tool calling (`src/agent/agent.py`) ; un client compatible OpenAI pour deux backends (`openrouter` : `openai/gpt-4.1-mini`, `temperature 0`, `seed 42` ; `ollama` : `qwen3:4b`). Réponse finale via un pseudo-outil `final_answer` validé par Pydantic. Configuration dans `configs/agent.yaml`, secret `OPENROUTER_API_KEY` dans `.env` uniquement (`.env.example` ajouté).
- **Prompts** : `prompts/system.md.j2` + `final_answer.md.j2` (Markdown gabarisé Jinja2, `StrictUndefined`), descriptions d'outils dans `prompts/tool_descriptions.yaml`. Seules des variables de configuration sont rendues ; la question et les sorties d'outils restent dans des messages séparés. `prompt_version` (SHA-256) est enregistré dans chaque résultat.
- *(Remplacé le 17 octobre par le contrat commun `AssistantResponse`, voir ci-dessus.)* **Contrat de sortie** `AgentResult` : réponse, citations (titre/URL remplis depuis le corpus), citations retirées, documents récupérés, appels d'outils, tokens, latence, coût (`usage.cost` d'OpenRouter), raison d'arrêt, abstention, `injection_suspected`, `ungrounded_numbers`. C'est la ligne attendue par `eval/run_eval.py`.
- **Évaluation** (`eval/eval_agent.py`, retiré le 17 octobre au profit de `eval/run_eval.py` ; golden v1, 33 questions, prompt `4757880a1880`) : routage exact **0,91** (rappel 1,00, précision 0,91) ; documents attendus récupérés **0,87** ; citations inventées **0** ; réponses avec chiffres non ancrés **0** ; abstention sur questions sans réponse **0,67** ; `final_answer` atteint **100 %** ; latence moyenne **4,5 s** (p95 6,4 s) ; ~5 550 tokens en entrée et ~190 en sortie par question ; coût total **0,047 $**.
- **Analyse d'erreurs** :
  1. *Département inventé (Saint-Denis → « 93 »)* : gpt-4.1-mini complète l'argument `departement` même quand la question ne le donne pas ; une consigne de prompt ne suffit pas. Corrigé en Python : un département absent de la question est retiré si la commune est homonyme, et l'outil renvoie les candidats.
  2. *Bug de ponctuation (q027)* : « Saint-Denis, en Seine-Saint-Denis, … » — la virgule empêchait de reconnaître le département. Corrigé + test de non-régression.
  3. *Dates inventées (q003, q004, q006)* : « ces informations sont valables en 2024 », détecté par le contrôle d'ancrage des chiffres. Corrigé par une consigne explicite ; 0 occurrence ensuite.
  4. *Abstention sous-déclarée (q019, q032)* : la réponse explique bien l'absence de données (disponibilité, loyers), mais avec `abstained=false`, et q032 met en avant des prix de vente. Consigne précisée ; reste 2 cas.
  5. *Routage « en trop » (q019, q032, q033)* : le modèle appelle RPLS/DVF pour donner du contexte alors que le golden set attend 0 outil. Comportement défendable : à arbitrer (étiquettes du golden set vs consigne).
  6. *Homonymie Montreuil (q023)* : Montreuil existe dans 4 départements ; l'agent demande la précision au lieu de supposer le 93 attendu par le golden set. Choix de sûreté (une mauvaise zone donnerait de mauvais plafonds), à arbitrer.
  7. *Retrieval (q005, q017, q025)* : la fiche F869 (plafonds, très longue, tableaux aplatis) ne remonte pas, même avec k = 5 sur les requêtes reformulées par le modèle. Point à traiter côté retrieval/chunking.
- **Sécurité** : à l'écriture des expressions régulières, des caractères invisibles (U+200B…U+FEFF) s'étaient glissés littéralement dans `guards.py` ; réécrits en séquences d'échappement et contrôle ajouté.
- **Tests** : 145 tests hors intégration (dont 20 pour l'agent avec un LLM scripté, 52 de sécurité) + 1 test d'intégration OpenRouter (`uv run pytest -m integration`).
- **Non fait** : backend Ollama non exécuté (Ollama absent de la machine d'Alexis) ; `eval/run_eval.py` et le jugement de la justesse des réponses restent à faire.

### 13 octobre 2026 — outils déterministes (branche `alexis`)

- **Configuration** : paramètres non secrets dans `configs/tools.yaml`, chemins fixes dans `src/config.py`, réglages typés dans `src/settings.py` (pydantic-settings : YAML < `.env` < variables `HLM_*`). `src/retrieval/config.py` n'est pas modifié.
- **Référentiel des communes** : COG Insee 2026 → `data/processed/communes_ref.csv` (34 875 communes, 45 arrondissements, 2 576 communes déléguées/associées en alias). Le résolveur (`src/tools/communes.py`) gère accents, « St », homonymes (erreur `AmbiguousCommune` avec candidats), arrondissements et codes Insee.
- **`check_eligibility`** : barème 2026 (arrêté du 19 décembre 2025, JO du 24 décembre 2025), 3 zones × PLAI/PLUS/PLS × catégories 1–6 + personne supplémentaire. Les catégories de ménage suivent l'arrêté du 29 juillet 1987 ; les définitions ne se cumulent pas (au plus n + 1). DOM non couverts (`DataNotCovered`).
- **Constat** : les montants affichés par Service-Public F869 (ex. 34 996 € pour 1 personne à Paris) sont les plafonds **PLS**, pas PLUS. Contrôles : PLS = round(1,3 × PLUS) sur les 21 valeurs, et tous les plafonds PLS saisis figurent dans le texte de F869 du corpus (test automatique).
- **Cas réel** : Pierrefitte-sur-Seine est une commune déléguée de Saint-Denis (93066) depuis 2025 ; elle hérite de la zone « Paris et communes limitrophes », avec un avertissement.
- **RPLS** : le seul fichier public est le fichier détaillé au logement (5 416 826 logements au 1er janvier 2025, téléchargé via DiDo avec 10 colonnes). Il ne contient **ni vacance, ni loyers, ni occupation** : l'outil donne la composition du parc (financement initial regroupé, type, taille, surface, part en QPV, ancienneté, DPE). 16 846 communes, 0 code orphelin. Ex. : Créteil 15 133 logements, Paris 250 640.
- **DVF** : geo-dvf 2024 et 2025, France entière, agrégé avec DuckDB en ~6 s. Seules les ventes d'un logement unique sont retenues (715 750 en 2024, 808 825 en 2025) ; 554 ventes écartées (6 codes communes absents du COG). Ex. 2025 : Paris appartements 9 700 €/m², Lyon 4 467 €/m², Créteil 3 634 €/m². Alsace-Moselle et Mayotte : `DataNotCovered`.
- **Tests** : 88 nouveaux tests (outils + sécurité des entrées), 110/110 avec le retrieval.
- **Prochaine action** : agent (OpenRouter + Ollama), registre d'outils, garde-fous, prompts Jinja2.

### 13 octobre 2026 — retrieval final : hybride RRF + reranker, `search_docs`

- Socle commun créé : `src/retrieval/config.py` (toute la configuration), `common.py`, `bm25.py` (`BM25Retriever`, tokenisation identique à `eval_bm25_chunking.py`) et `eval/metrics.py` (chunks → documents, Recall@1/5/10, Hit@1, MRR).
- Cache des embeddings denses dans `data/indexes/` (`.npy` + JSON : modèle, nombre de chunks, SHA-256 du fichier de chunks ; reconstruit automatiquement si l'un change ; ignoré par Git) : calcul **17,9 s**, rechargement **0,01 s**, embeddings identiques au bit près.
- Non-régression validée : BM25 **1,0000 / 1,0000 / 0,8306** et dense E5 **0,9444 / 1,0000 / 0,7394** (R@5 / R@10 / MRR), avec des rangs par question identiques aux anciens CSV.
- Hybride (`src/retrieval/hybrid.py`) : RRF au niveau des chunks, score = Σ 1/(60 + rang), 100 candidats par retriever ; `bm25_rank`, `dense_rank` et `rrf_score` sont conservés pour la traçabilité.
- Reranker (`src/retrieval/reranker.py`) : cross-encoder appliqué aux 30 premiers candidats, entrée (question, `retrieval_text`), MPS, `max_length` 512 (pour q001, 178 paires sur 1 357, soit 13 %, dépassent 512 tokens et sont tronquées). Modèles comparés : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118 M paramètres, 471 Mo) et `BAAI/bge-reranker-v2-m3` (568 M paramètres, 2 271 Mo).
- Ablation `eval/eval_retrieval.py` → `eval/retrieval/summary.csv` et `eval/retrieval/details.csv` :

  | Méthode | R@1 | R@5 | R@10 | MRR | Latence moy. | p95 | Chargement |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | BM25 | 0,6944 | 1,0000 | 1,0000 | 0,8306 | 2,3 ms | 3,1 ms | 0,10 s |
  | Dense E5 | 0,5833 | 0,9444 | 1,0000 | 0,7394 | 9,5 ms | 12,9 ms | 3,94 s |
  | Hybride RRF | 0,7500 | 0,9444 | 1,0000 | 0,8524 | 10,5 ms | 11,3 ms | 4,04 s |
  | Hybride + MiniLM | 0,8611 | 1,0000 | 1,0000 | 0,9444 | 452,4 ms | 502,7 ms | 6,74 s |
  | Hybride + bge | 0,9167 | 1,0000 | 1,0000 | 0,9722 | 4 281,7 ms | 4 427,8 ms | 8,99 s |

- 9 questions sur 18 ont un rang différent selon la méthode. L'hybride seul améliore le MRR de BM25 (0,8306 → 0,8524) mais fait sortir **q006 / F21051** du top 5 (rang 7, comme le dense) ; les deux rerankers la remettent au rang 1. Avec MiniLM, deux questions restent au rang 2 : q005 (F869) et q013 (F31864) ; avec bge, seulement q013.
- Choix selon la règle fixée avant l'expérience : bge a le meilleur MRR (écart +0,0278, au-dessus du seuil de bruit de 0,02), mais sa latence p95 (**4 428 ms**) dépasse 2 s sur le Mac → **MiniLM retenu par défaut** (MRR 0,9444, p95 503 ms). bge reste activable via `DEFAULT_RERANKER` dans `config.py`.
- Interface `search_docs(query, k=5, source=None, rerank=True) -> SearchResponse` (Pydantic v2, `src/retrieval/search.py` et `schemas.py`) : chargement unique (lazy singleton) et `warmup()`, filtre `source` appliqué avant la coupe à k, textes renvoyés signalés comme des données. Démonstration : `uv run python -m src.retrieval.search "Comment faire un recours Dalo ?"` (chargement 7,6 s, requête 694 ms).
- Tests `tests/test_retrieval.py` : **22 tests passés** (`uv run pytest -q`, 21 s).
- Environnement : `uv` a recréé `.venv` avec **Python 3.14.4** (aucun `.python-version` dans le dépôt), PyTorch 2.14.1 et sentence-transformers 6.1.0, MPS disponible ; les résultats BM25 et dense sont inchangés. Premier téléchargement des deux rerankers : environ 17 min.
- Limites : 18 questions seulement (une question pèse 0,056 sur le Hit@1) ; aucune optimisation d'hyperparamètres (valeurs standard) ; latences mesurées sur un seul passage de l'évaluation.
- Prochaine action : intégrer `qwen3:4b` via Ollama et créer `eval/run_eval.py`.


### 13 octobre 2026 — Ollama et modèle local validés

- Machine locale confirmée : **MacBook Pro M1, 16 Go de mémoire unifiée** ; Homebrew est installé.
- Ollama installé et opérationnel ; `ollama list` confirme **`qwen3:4b`** (2,5 Go).
- Premier test interactif exécuté avec `ollama run qwen3:4b`.
- Le test sur « DALO » montre que le modèle seul ne connaît pas fiablement ce terme juridique français et produit une réponse erronée : cela confirme la nécessité de fournir les connaissances métier via le **RAG Service-Public + LEGI**, plutôt que de compter sur la mémoire interne du LLM.
- Le modèle `qwen3:4b` expose un mode de raisonnement (« thinking »), à mesurer séparément car il augmente les tokens et la latence.
- Prochaine étape LLM : tester l'API locale Ollama depuis le projet Python, puis intégrer le modèle au pipeline RAG après la recherche hybride et le reranker.


### 11 octobre 2026 — dense retrieval versionné et dépôts synchronisés

- Dépendance `sentence-transformers` ajoutée ; environnement local validé avec **PyTorch 2.14.1** et **MPS disponible**.
- `src/retrieval/dense.py`, `eval/eval_dense.py`, `eval/dense/details.csv` et `eval/dense/summary.csv` créés et versionnés.
- Analyse d'erreurs dense : q003 rang 2, q004 rang 4, q005 rang 2, **q006 rang 7**, q013 rang 4, q014 rang 3, q015 rang 3.
- Résultats denses confirmés : **Recall@5 = 0,9444**, **Recall@10 = 1,0000**, **MRR = 0,7394** ; BM25 reste supérieur seul (**1,0000 / 1,0000 / 0,8306**).
- Commit dense après rebase/push : **`8f22ce1f71bdd46f5b85047a7632018ea85660a8`**.
- Vérification finale : **LOCAL = ORIGIN = REVIEW = 8f22ce1f71bdd46f5b85047a7632018ea85660a8**.
- Prochaine action : implémenter et évaluer la fusion hybride **BM25 + dense par RRF**.


### 11 octobre 2026 — dense retrieval E5 évalué

- Modèle dense évalué : `intfloat/multilingual-e5-small`, dimension **384**, exécuté sur **MPS**.
- Corpus dense : **1 357 chunks** issus du chunking de référence `fixed_300_60`.
- Temps de construction de l'index dense : **23,46 s** ; latence moyenne de requête : **33,37 ms/question**.
- Résultats sur les **18 questions RAG** du golden set : **Recall@5 = 0,9444**, **Recall@10 = 1,0000**, **MRR = 0,7394**.
- Baseline BM25 sur le même chunking : **Recall@5 = 1,0000**, **Recall@10 = 1,0000**, **MRR = 0,8306**.
- Conclusion provisoire : le dense E5 seul est moins performant que BM25 sur ce golden set, mais conserve un rappel parfait à 10 ; prochaine étape = analyse d'erreurs par question puis construction d'une recherche hybride BM25 + dense.


### 11 octobre 2026 — chunking de référence figé après analyse d'erreurs BM25

- Analyse des rangs BM25 réalisée sur les trois chunkings.
- `fixed_300_60` conserve **Recall@5 = 1,0000** et **Recall@10 = 1,0000** ; 5 questions ont leur premier document pertinent après le rang 1, mais aucune ne sort du Top 5. Cas observés : q005 rang 2, q006 rang 2, q007 rang 2, q008 rang 4, q013 rang 5.
- `paragraph_400_80` conserve également un rappel parfait à 5/10, mais avec un **MRR inférieur (0,8074)** et 6 questions dont le premier document pertinent n'est pas classé premier.
- `fixed_500_100` est moins performant : q008 et q013 sortent du Top 5 (premiers documents pertinents aux rangs 8 et 6), d'où **Recall@5 = 0,8889** et **MRR = 0,7106**.
- Décision : **`fixed_300_60` devient le chunking de référence** pour la suite des expériences, car il obtient le meilleur MRR tout en gardant un rappel parfait à 5 et 10.
- Prochaine action : versionner le script et les sorties de l'expérience BM25, puis implémenter le dense retrieval sur le même chunking de référence.


### 11 octobre 2026 — comparaison BM25 des trois chunkings

- Évaluation BM25 exécutée sur les **18 questions RAG** du golden set avec les trois stratégies de chunking.
- `fixed_300_60` : **Recall@5 = 1,0000**, **Recall@10 = 1,0000**, **MRR = 0,8306**.
- `fixed_500_100` : **Recall@5 = 0,8889**, **Recall@10 = 1,0000**, **MRR = 0,7106**.
- `paragraph_400_80` : **Recall@5 = 1,0000**, **Recall@10 = 1,0000**, **MRR = 0,8074**.
- Interprétation provisoire : `fixed_300_60` est la meilleure baseline lexicale des trois, avec un rappel parfait à 5/10 et le meilleur MRR. `paragraph_400_80` est très proche ; `fixed_500_100` perd en précision de classement.
- Le choix final du chunking n'est pas encore figé : prochaine action = analyser les rangs par question et les erreurs de `fixed_500_100`, puis décider si `fixed_300_60` devient le chunking de référence avant le dense retrieval.


### 11 octobre 2026 — trois stratégies de chunking générées

- Le script `src/preprocessing/build_chunks.py` a été exécuté sur les **455 documents** du corpus.
- `fixed_300_60` : **1 357 chunks**, 2,98 chunks/document en moyenne, moyenne **243,8 mots**, médiane **300**, maximum **300**.
- `fixed_500_100` : **944 chunks**, 2,07 chunks/document en moyenne, moyenne **345,0 mots**, médiane **500**, maximum **500**.
- `paragraph_400_80` : **1 175 chunks**, 2,58 chunks/document en moyenne, moyenne **295,8 mots**, médiane **371**, p95 **399**, maximum **400**.
- Les trois variantes couvrent les **455 documents**, avec **0 chunk vide** et **0 doublon de `chunk_id`**.
- Ces statistiques ne suffisent pas à choisir une stratégie : le choix sera fait sur les performances de retrieval.
- Prochaine action : construire une baseline BM25 identique pour les trois jeux de chunks et comparer **Recall@5**, **Recall@10** et **MRR** sur les **18 questions RAG** du golden set.


### 11 octobre 2026 — audit manuel du chunking terminé

- L'échantillon stratifié de **20 documents** (10 Service-Public, 10 LEGI) a été relu.
- **Service-Public** : les titres et sous-titres sont majoritairement conservés comme lignes séparées ; les documents longs comportent de nombreux paragraphes exploitables, mais contiennent aussi des glossaires, références et liens connexes en fin de fiche.
- **LEGI** : les articles courts tiennent souvent en un seul paragraphe ; les articles plus longs sont structurés en alinéas/paragraphes numérotés mais disposent de moins de titres internes dans le texte lui-même.
- Décision : conserver les petits documents entiers et comparer trois stratégies reproductibles : **fixed_300_60**, **fixed_500_100** et **paragraph_400_80** (paragraphes respectés autant que possible).
- Chaque chunk devra conserver au minimum `chunk_id`, `document_id`, `source`, `title`, position/ordre du chunk et texte.
- Prochaine action : générer les trois corpus de chunks, auditer leur distribution puis les comparer avec une baseline BM25 sur le golden set.


### 11 octobre 2026 — échantillon de 20 documents pour le chunking

- Échantillon stratifié de **20 documents** généré dans `data/processed/chunking_sample_20.md`.
- Répartition : **10 Service-Public** et **10 LEGI**, couvrant volontairement toute la distribution de longueur.
- Service-Public observé dans l'échantillon : de **8 à 7 498 mots** ; LEGI : de **12 à 2 868 mots**.
- L'échantillon est prêt pour la lecture manuelle des séparations de paragraphes, titres/sous-titres et gros blocs avant de figer les stratégies de chunking.
- La tâche « lire manuellement au moins 20 documents » reste ouverte tant que cette lecture n'a pas été confirmée.


### 10 octobre 2026 — corpus RAG combiné finalisé

- Script `src/preprocessing/build_rag_corpus.py` exécuté avec succès.
- Fusion des **117 documents Service-Public** et des **338 articles LEGI**.
- Corpus combiné final : **455 documents**.
- Contrôles qualité : **0 identifiant vide**, **0 texte vide**, **0 titre vide**, **0 doublon de `document_id`**.
- Répartition par source : **338 LEGI**, **117 SERVICE_PUBLIC**.
- Longueurs observées : minimum **8 mots**, maximum **7 498 mots**, moyenne **608,2 mots**.
- Fichiers générés : `data/processed/rag_documents.jsonl` et `data/processed/rag_documents_audit.csv`.
- Prochaine action : définir et comparer plusieurs stratégies de chunking documentaire avant l'implémentation de BM25.


### 10 octobre 2026 — corpus LEGI HLM final généré

- Script `src/preprocessing/build_legi_hlm_corpus.py` exécuté avec succès.
- Périmètre source : **4 559 lignes/chunks VIGUEUR**, correspondant à **4 047 `doc_id` uniques**.
- Périmètre juridique ciblé : **373 lignes/chunks**, correspondant à **338 articles uniques**.
- Reconstruction validée : **338 documents**, **0 vide**, **0 doublon de `doc_id`**, **338 numéros d'article uniques**.
- **27 articles** du périmètre final étaient multi-chunks et ont été reconstruits dans l'ordre de `chunk_index`.
- Répartition : **159 articles R**, **143 articles L**, **36 articles D**.
- Longueurs : minimum **12 mots**, maximum **2 868 mots**, moyenne **209,1 mots**.
- Fichiers générés : `data/processed/legi_hlm_documents.jsonl` et `data/processed/legi_hlm_documents_audit.csv`.
- Avec les **117** fiches Service-Public déjà figées, le corpus documentaire combiné attendu est de **455 documents** avant chunking.
- Prochaine action : fusionner Service-Public et LEGI dans un corpus documentaire commun, effectuer les contrôles finaux, puis commencer les expérimentations de chunking.


### 10 octobre 2026 — diagnostic des chevauchements LEGI

- **512 paires** de chunks successifs ont été analysées sur les articles LEGI multi-chunks.
- **505 paires** ne présentent aucun chevauchement exact ; **7 paires** seulement présentent un chevauchement, avec un maximum de **18 caractères**.
- Les chevauchements de 1 caractère observés sont compatibles avec de simples coïncidences de frontière ; les autres restent très courts (4, 11, 16 et 18 caractères).
- Décision de reconstruction : concaténer les morceaux `text` dans l'ordre de `chunk_index` en conservant intégralement le texte, plutôt que supprimer automatiquement des suffixes/préfixes et risquer d'altérer un texte juridique.
- Prochaine action : générer `data/processed/legi_hlm_documents.jsonl` avec **1 `doc_id` = 1 article reconstruit**, puis contrôler vides, doublons et couverture du périmètre cible.


### 10 octobre 2026 — diagnostic de reconstruction des articles LEGI

- Diagnostic exécuté sur les articles `VIGUEUR` du CCH : **232 articles** sont répartis sur plusieurs chunks.
- Un seul article a un champ `text` dont la version la plus longue contient toutes les autres ; **231 articles nécessitent donc une reconstruction ordonnée**.
- Les exemples les plus fragmentés confirment que `text` contient des morceaux successifs de l'article alors que `chunk_text` ajoute surtout le contexte hiérarchique et le titre.
- Maximum observé : **83 chunks** pour `LEGIARTI000054024988` (annexe à l'article R172-4).
- Prochaine action : mesurer les chevauchements entre morceaux `text` successifs, puis reconstruire les articles en concaténant les chunks dans l'ordre sans duplication.


### 10 octobre 2026 — audit LEGI au niveau article

- Le Parquet du Code de la construction et de l'habitation a été audité au niveau des articles en vigueur.
- **4 559 lignes/chunks VIGUEUR** correspondent à **4 047 `doc_id` uniques**.
- **232 articles** sont répartis sur plusieurs chunks, avec un maximum observé de **83 chunks pour un article**.
- Pour ces 232 articles, le champ `text` varie entre les chunks : il ne faut donc **pas** conserver arbitrairement la première ligne par `doc_id`.
- Le périmètre juridique ciblé contient **338 articles uniques** : **336** du Titre IV « Rapports des organismes HLM et des bénéficiaires », plus `L300-1` (DALO) et `L411` (socle général HLM).
- Répartition des 338 candidats : **159 R**, **143 L**, **36 D** ; **0 document vide** et **0 doublon de `doc_id`** dans l'audit.
- Avec les **117** fiches Service-Public déjà figées, ce périmètre conduirait à **455 documents** avant chunking, ce qui est cohérent avec la cible d'environ 500 documents.
- Prochaine action : diagnostiquer précisément la relation entre `text` et `chunk_text` pour reconstruire correctement chaque article multi-chunk avant génération du corpus LEGI final.


### 10 octobre 2026 — accès LEGI via AgentPublic/legi validé

- L'accès direct aux pages Légifrance par `curl` a été abandonné après réponses HTTP **403**.
- La source `AgentPublic/legi` a été testée avec `datasets.load_dataset(..., streaming=True)`.
- Le dataset expose bien les champs nécessaires au projet : `doc_id`, `nature`, `category`, `status`, `title`, `number`, `start_date`, `end_date`, `subtitles` et `text`.
- Cinq articles du **Code de la construction et de l'habitation** ont été lus avec succès ; le test confirme aussi la présence de plusieurs versions historiques (`MODIFIE`, `ABROGE`).
- Le fichier Parquet dédié au CCH a ensuite été identifié et téléchargé : **1 fichier, 89,2 Mo**.
- Le Parquet contient **16 599 lignes/chunks** au total, dont **4 559 lignes `VIGUEUR`** ; le premier filtre structurel HLM/DALO retourne **938 lignes candidates**.
- Point important : `AgentPublic/legi` est déjà chunké (`chunk_id`, `chunk_index`, `chunk_text`, embeddings). Pour respecter notre propre expérimentation de chunking, le corpus LEGI doit être reconstruit au niveau **document/article** à partir de `doc_id` et `text`, en ignorant les chunks et embeddings fournis.
- Prochaine action : mesurer le nombre de `doc_id` uniques, dédupliquer les lignes par article, puis resserrer le périmètre sur le Titre IV HLM et le socle DALO avant de générer le corpus LEGI final.


### 9 octobre 2026 — corpus Service-Public HLM final figé

- Revue manuelle du périmètre terminée.
- Les **33 documents `REVIEW`** ont été reclassés hors du cœur HLM.
- Les **13 documents conservés uniquement par `social_keyword`** ont été audités : **5 conservés**, **8 exclus**.
- Script `src/preprocessing/build_service_public_hlm_corpus.py` exécuté localement.
- Corpus final généré dans `data/processed/service_public_hlm_documents.jsonl`.
- Décisions finales générées dans `data/processed/service_public_scope_decisions_final.csv`.
- Résultat final : **117 documents conservés**, **233 exclus**, **0 document vide**, **0 doublon**.
- Vérification du golden set : **15 gold_docs attendus, 15 présents, 0 manquant**.
- Prochaine action : intégrer LEGI avant le chunking documentaire et l'implémentation de BM25.


### 9 octobre 2026 — classification du périmètre et synchronisation des dépôts

- Fichier `data/processed/service_public_scope_decisions.csv` généré.
- Répartition automatique : **125 KEEP**, **33 REVIEW**, **192 EXCLUDE**.
- Raisons principales : 97 `core_dossier`, 15 `gold_doc`, 13 `social_keyword`, 26 `related_subtheme`, 7 `very_short`, 192 `outside_hlm_scope`.
- Vérification du golden set : **15 gold_docs attendus, 0 manquant, tous conservés**.
- Les dépôts `origin/main` et `review/main` sont synchronisés sur le commit `cc2103723f64e55a2d402fc08b35509690fc2807`.
- Prochaine action : revue manuelle des 33 documents `REVIEW`, puis génération du corpus Service-Public HLM final.


### 9 octobre 2026 — audit hiérarchique du périmètre Service-Public

- `data/processed/service_public_scope_audit.csv` créé pour les **350** fiches.
- Répartition par sous-thème : **107 Bail d’habitation**, **63 Urbanisme**, **47 Copropriété**, **40 Aides financières pour le logement**, **31 Vie pratique dans un logement**, **28 Achat immobilier**, **26 Protection et sécurité de l'habitat**, **8 Expropriation - Préemption**.
- Dossiers HLM directement pertinents identifiés : **Accéder à un logement social (8)** et **Aides personnelles au logement (8)**.
- Plusieurs dossiers de bail d'habitation sont également utiles pour les droits des locataires sociaux ; à l'inverse, urbanisme, copropriété, travaux, forage, piscines et sujets similaires sont hors cœur du projet.
- Prochaine action : appliquer une règle de sélection explicite et vérifiable, tout en forçant la conservation des `gold_docs` du golden set.


### 9 octobre 2026 — parser corrigé, corpus audité et versionné

- Correction ciblée de la déduplication : les conteneurs `ASavoir`, `Attention`, `ANoter`, `Exemple` et `Definition` sont dédupliqués sans supprimer les paragraphes métier.
- Validation sur `F21051` : la phrase expliquant le surloyer et le seuil de **20 %** est conservée.
- Extraction complète relancée : **350 documents**, **0 vide**, minimum **48 caractères / 8 mots**, maximum **45 331 caractères / 7 498 mots**, moyenne **9 872,5 caractères / 1 555,6 mots**.
- Audit des 10 documents les plus courts et des 10 plus longs réalisé.
- Lecture d'un échantillon reproductible de **20 documents** avec `random.seed(42)`.
- Constat : le filtre `subject` contenant `Logement` est trop large pour un assistant HLM ; l'échantillon contient notamment taxe d'aménagement, forage, copropriété, piscine et urbanisme.
- Le parser a été commité et poussé sur `origin/main` et sur le dépôt `review` au commit `52d1539`.
- Prochaine action : enrichir les métadonnées hiérarchiques puis définir un périmètre Service-Public ciblé avant chunking.


### 9 octobre 2026 — première correction de redondance à ajuster

- Une première correction de `extract_text_blocks` a été testée sur les 5 fiches de référence pour éviter l'extraction simultanée des conteneurs et de leurs sous-blocs.
- La redondance visible a nettement diminué : par exemple les encadrés `À savoir` ne sont plus répétés sous forme conteneur + titre + paragraphe.
- Cependant, la règle était trop générale : elle supprimait aussi certains `Paragraphe` contenant un descendant textuel tel que `Valeur`. Sur `F21051`, cela a fait disparaître des phrases importantes contenant « 20 % », tout en conservant seulement la valeur isolée.
- Conclusion : ne pas figer cette version du parser. La déduplication doit s'appliquer uniquement aux balises conteneurs (`ASavoir`, `Attention`, `ANoter`, `Exemple`, `Definition`) et non aux paragraphes, items, titres ou valeurs.
- Prochaine action : appliquer cette correction ciblée, relancer les 5 fiches de référence et vérifier que le texte utile est conservé sans répétition structurelle.


### 9 octobre 2026 — parser Service-Public et première extraction complète

- Environnement local vérifié : Python **3.12.7**, dépendances importables avec `uv`.
- `src/ingestion/service_public.py` créé localement et compilé sans erreur de syntaxe.
- Test sur 5 fiches de référence : `F10007`, `F18005`, `F12006`, `F869`, `F21051`.
- Résultat du test : **5 documents**, **0 vide**, tous les identifiants attendus présents.
- Première extraction du corpus Logement : **350 documents**, **0 vide**.
- Longueurs observées : minimum **48 caractères / 8 mots**, maximum **54 224 caractères / 8 940 mots**, moyenne **12 029 caractères / 1 888,7 mots**.
- Répartition des sujets : 333 `Logement`, 13 `Logement, Secteurs d'activité`, 2 `Logement, Fiscalité`, 1 `Logement, Secteurs d'activité, Environnement`, 1 `Argent - Impôts - Consommation, Logement`.
- Le contrôle visuel montre des répétitions sémantiques dues à l'extraction simultanée de conteneurs et de leurs sous-blocs (par exemple un encadré `À savoir` puis son titre/paragraphe). Le contrôle « doublons stricts = 0 » ne suffit donc pas.
- Prochaine action : corriger cette redondance structurelle, relancer l'extraction, puis auditer les documents extrêmes avant de figer le corpus.


### 8 octobre 2026 — migration et structuration GitHub

- Dépôt de référence : `Charles-Scannavino/Projet_Generative_AI`.
- Fichiers réorganisés : documentation dans `Docs/`, notebook dans `notebooks/`, golden set dans `eval/`, corpus brut Service-Public dans `data/raw/service_public/`.
- Création des répertoires de travail `src/`, `app/`, `data/processed/`, `data/indexes/` et des répertoires bruts futurs.
- Golden set actif porté à **20 questions** ; backlog séparé pour éligibilité/RPLS/DVF.
- README et `.gitignore` adaptés au projet.
- GitHub devient la source de vérité pour le code, le notebook, la documentation et l'évaluation.


### 8 octobre 2026 — analyse de la dernière version du notebook

- Notebook `Assistant_HLM_RAG (1).ipynb` analysé : **41 cellules**.
- Extraction locale Colab confirmée : **5 574 XML**.
- Parsing des `F*.xml` confirmé : **3 004 fiches**.
- Filtre HLM strict : **24** fiches.
- Filtre `subject` contenant `Logement` : **350** fiches.
- Filtre logement élargi par mots-clés : **505** fiches, avec faux positifs visibles.
- Corpus Service-Public de travail fixé provisoirement à **350 fiches** du sujet `Logement`.
- Sauvegarde confirmée : `data/processed/service_public_logement_metadata.csv`.
- Golden set actuel du notebook : **5 questions de brouillon**, encore incomplètes pour les réponses/sources.
- Prochaine action : extraction et contrôle du texte complet des 350 fiches.

### 8 octobre 2026 — adaptation du suivi projet

- Relecture et adaptation des anciens fichiers `CONTEXT.md` et `AVANCEMENT.md`.
- Suppression du Transformer comme contexte actif : il reste un historique séparé et n'est plus le cœur de ce projet.
- Recentrage complet sur l'Assistant HLM RAG & Agent.
- Mise en cohérence avec le travail réellement effectué dans Google Colab.
- Intégration de l'état actuel du corpus Service-Public et du résultat de 24 fiches HLM candidates.
- Clarification : le chunking documentaire n'est pas encore réalisé.
- Ajout de la règle de mise à jour régulière de ce fichier.

### 15 octobre 2026 — golden set v2, run_eval, agent et interface (Gaby)

- Golden set porté à **52 questions** avec deux splits : `dev` (q001-q020, ayant servi aux choix du retrieval, texte inchangé) et `test` (q021-q052, **held-out**).
- Nouveaux types : 10 questions en langage courant (dont 6 « jumeaux » de questions dev, champ `twin_of`), 7 questions sur les articles LEGI (0 auparavant), 5 cas d'éligibilité calculés à la main depuis l'arrêté du 19/12/2025, 2 demandes de précision, 2 RPLS/DVF (`pending_data`), 2 hors périmètre, 2 injections, 2 pièges causaux.
- Nouveaux champs : `split`, `acceptable_docs` (documents qui contiennent aussi la réponse, comptés pour Hit@k/MRR), `expected_tool`, `should_abstain`, `expected`.
- `eval/check_golden.py` : intégrité (ids, types, documents présents dans le corpus). BM25 seul : Hit@5 = 1,00 sur `dev` (18 questions) mais **0,47 sur `test`** (17 questions) ; le split `test` discrimine enfin les méthodes de retrieval.
- Constat données : la fiche F869 affiche les plafonds **PLS** (PLUS x 1,3) et ses tableaux sont aplatis par le parser (montants sans libellé de catégorie) ; d'où un outil déterministe plutôt qu'une lecture par le LLM.
- `src/tools/eligibility.py` (+ 11 tests) : catégories de ménage, jeune ménage, handicap, 3 zones, PLAI / PLUS / PLS 2026.
- `src/agent/` : contrat `AssistantResponse` commun ; backends `baseline` (BM25 sans LLM), `rag` (extrait du notebook, `think=False`, tokens, citation inventée supprimée et signalée au lieu d'une exception) et `agent` (tool calling Ollama, validation Pydantic des arguments, au plus 3 tours).
- `eval/run_eval.py` : commande de l'énoncé, `results.jsonl` complet, `summary.md`, juge LLM optionnel, `--run-name` / `--compare` pour les ablations.
- `app/streamlit_app.py` : réponse, sources, calcul d'éligibilité, métriques, trace de l'agent, onglet Évaluation.
- Scripts d'évaluation du retrieval restreints au split `dev` par défaut (résultats de référence inchangés) ; `eval_retrieval.py test` pour le held-out.
- Constat modèle (16 octobre) : le tag Ollama `qwen3:4b` pointe vers Qwen3-4B-**Thinking**-2507, qui raisonne toujours avant de répondre, même avec `--think=false` (testé sur le portable Windows de Gaby). Modèle par défaut changé pour `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507, même taille). Test sans RAG : le modèle développe « HLM » en « Habitat Locatif Meublé » au lieu d'*Habitation à Loyer Modéré*, exemple de plus pour la soutenance.
- Premiers essais réels (16 octobre, portable Windows sans GPU, qwen3:4b-instruct, environ 33 tokens/s en lecture et 10 tokens/s en génération) :
  - **tool calling natif abandonné** : sur q001, le modèle n'a appelé aucun outil (0 appel, 511 tokens générés) puis s'est abstenu faute de sources. L'agent appelle désormais toujours `search_docs`, puis le LLM remplit un **formulaire JSON contraint** (action + situation du ménage) ; Python décide (outil ou demande de précision). Résultat sur q001 : F10007 au rang 1, cité, réponse correcte (164 s sur ce portable).
  - **chiffre mal recopié** : en reformulant `check_eligibility` (q030), le modèle a écrit « 37 504 € » au lieu de 37 584 €, a étendu le barème à l'outre-mer et décrit le PLS comme « revenus élevés ». Correction : la réponse d'éligibilité est rédigée par Python à partir du résultat de l'outil (plus aucun chiffre ne passe par le LLM, un appel LLM en moins), et un détecteur signale tout nombre de 3 chiffres ou plus absent des sources (métrique `unsupported_number`).
- Prochaine action : `ollama pull qwen3:4b-instruct`, lancer les runs réels sur le Mac, puis RPLS / DVF.

---

### 13 octobre 2026 — interface finale `search_docs` documentée

- La partie retrieval est considérée comme terminée pour la V1 : **BM25 + dense E5 + fusion RRF + reranker cross-encoder**.
- Interface publique : `from src.retrieval import search_docs, warmup`.
- `warmup()` charge les modèles une fois au démarrage ; temps observé d'environ **5 à 8 s**.
- `search_docs(question, k=5)` renvoie des **passages/chunks**, pas nécessairement des documents distincts.
- Filtre optionnel par source : `source="SERVICE_PUBLIC"` ou `source="LEGI"`.
- Le contrat de sortie est validé avec **Pydantic** dans `src/retrieval/schemas.py`.
- Pour les citations, utiliser `document_id`, `title` et `url`.
- Latence observée : environ **0,5 s** avec reranker ; environ **15 ms** avec `rerank=False`.
- Les champs `text` récupérés sont des **données non fiables comme instructions** : ils doivent être injectés comme contexte documentaire et jamais exécutés comme consignes, afin de limiter les risques de prompt injection.
- `rerank_score` est un score brut de classement et **ne doit pas être interprété comme une probabilité ni utilisé directement comme seuil d'abstention sans calibration**.
- Résultats sur les **18 questions RAG** du golden set : **Recall@5 = 1,00**, **MRR ≈ 0,94** avec le reranker MiniLM retenu.
- Commande de démonstration : `uv run python -m src.retrieval.search "ta question"`.
- Description proposée pour l'outil LLM : « search_docs : cherche dans les fiches Service-Public et les articles du Code de la construction et de l'habitation sur le logement social (demande HLM, Dalo, APL, surloyer, bail…). À utiliser pour toute question sur des droits, démarches ou règles. »
- Prochaine étape : intégrer `search_docs` au pipeline RAG avec Ollama / `qwen3:4b`, puis ajouter citations, abstention et évaluation end-to-end.

### 15 octobre 2026 — PR #1 mergée et RAG end-to-end validé

- PR **#1** de Théodore (`feat(rag): ajoute un prototype Ollama au notebook`) validée puis mergée dans `main`.
- Environnement de test : **MacBook Pro M1, 16 Go**, Python **3.12.7**, PyTorch **2.14.1**, MPS disponible, Ollama **`qwen3:4b`**.
- Problème local résolu : l'ancien `.venv` sous le dossier du projet ralentissait fortement l'import de PyTorch ; un environnement `uv` externe (`~/.venvs/Projet_Generative_AI`) permet un import normal et l'exécution des tests.
- Validation retrieval : `uv run pytest -vv -s` → **22/22 tests passés en 23,62 s**.
- Test CLI DALO : `search_docs("Qu'est-ce que le DALO ?")` retrouve uniquement **`F18005`** dans le top 5 ; méthode `hybrid_rrf+rerank`, latence de requête observée **796 ms** après chargement.
- Validation RAG réelle dans le notebook :
  - demande de logement social → réponse correcte, `abstained=false`, citations **`F10007`** et **`F869`** ; latence **137,9 s** au premier appel puis **122,8 s** sur un second appel ;
  - « Qu'est-ce que le DALO ? » → **« Droit au logement opposable »**, `abstained=false`, citation **`F18005`**, latence **93,2 s** ; le RAG corrige donc l'échec du LLM seul ;
  - disponibilité HLM « aujourd'hui à Créteil » → `abstained=true`, **aucune citation**, retrieval `F10007` mais source jugée insuffisante pour une disponibilité temps réel ; latence **67,4 s**.
- Conclusion : pipeline **retrieval → Ollama/Qwen3 → Pydantic → citations/abstention** fonctionnel. Limite principale : latence générative encore élevée ; la sortie d'abstention utilisateur est aussi trop minimale (`"abstained"`) et doit être reformulée proprement.
- Prochaine action : extraire le composant RAG du notebook vers `src/`, mesurer tokens/latence par étape, optimiser Qwen/Ollama, puis implémenter `eval/run_eval.py`.

### 13 octobre 2026 — prototype RAG ajouté au notebook

- Les dernières modifications de `origin/main` ont été intégrées à la branche de travail avant l'ajout du notebook.
- Ajout d'une section RAG en fin de `notebooks/Assistant_HLM_RAG.ipynb`, après l'exploration/préparation Service-Public existante.
- Le prototype appelle le `search_docs` final du dépôt, envoie les passages récupérés à `qwen3:4b` via l'API HTTP locale Ollama et structure la sortie avec Pydantic.
- Les identifiants de citations sont contrôlés contre les documents effectivement récupérés ; le prompt traite le texte des sources comme des données, pas comme des instructions, et demande l'abstention lorsque les sources sont insuffisantes.
- Contrôles initiaux : JSON du notebook valide et espaces Git propres. Le runtime Ollama n'avait pas encore été lancé dans cette session ; cette validation a été réalisée le 15 octobre.

*Dernière mise à jour : 15 octobre 2026 — golden set v2 (52 questions), run_eval, agent et interface.*

### 18 octobre 2026 — serveur MCP et tests validés

**Responsable : Charles — bonus MCP.**

- Branche de travail : `charles-bonus-mcp`, créée depuis `main`.
- SDK MCP Python 2.3.0 installé avec `uv`.
- Serveur créé : `src/mcp_server.py`.
- Premier outil exposé : `search_docs`.
- Réutilisation directe du retrieval existant : BM25 + E5 + RRF + MiniLM.
- Réponse MCP structurée en JSON avec identifiants, sources, passages et latence.
- Découverte de l'outil validée avec un client MCP.
- Recherche DALO validée : document `F18005` retrouvé.
- Transport `stdio` validé avec un serveur exécuté en sous-processus.
- Tests automatisés ajoutés dans `tests/test_mcp_server.py`.
- Tests MCP : 2/2 réussis en 4,35 s.
- Tests globaux : 24/24 réussis en 19,15 s.

**Limites actuelles :**
- Seul `search_docs` est exposé.
- Pas encore de démonstration dans un client MCP tiers.
- Les autres outils seront intégrés après validation de leurs modules.
- Aucun commit ni push effectué à cette étape.

**Prochaines actions :**
1. Compléter les tests de validation des arguments et des erreurs.
2. Connecter le serveur à un client MCP externe.
3. Documenter l'installation et l'utilisation du serveur.
4. Préparer la livraison de la branche bonus MCP.


### 18 octobre 2026 — validations complémentaires MCP

- Gestion des erreurs avec `ToolError`.
- Question vide rejetée avec un message explicite.
- Valeur `k=0` rejetée avec un message explicite.
- Absence de traceback inattendue pour ces erreurs.
- Tests MCP : 4/4 réussis en 4,30 s.
- Tests globaux : 26/26 réussis en 19,11 s.
- Aucun échec détecté dans les tests existants.

**Prochaine étape :** premier commit sur la branche dédiée,
puis démonstration et documentation du client MCP externe.

### 18 octobre 2026 — MCP Inspector : démonstration externe validée

- Branche : `charles-bonus-mcp`.
- Premier commit publié : `9144590`.
- Client externe : MCP Inspector 2.9.0.
- Connexion au serveur `assistant-hlm` via `stdio` : validée.
- Outil découvert et exécuté : `search_docs`.
- Question : « Qu'est-ce que le DALO ? »
- Nombre de passages demandé : 5.
- Nombre de passages retourné : 5.
- Document retrouvé : `F18005` (Service-Public.fr).
- Méthode : `hybrid_rrf+rerank`.
- Latence observée : 9 068,57 ms.
- Réponse structurée : identifiants, sources, URLs, textes et scores.
- Le serveur MCP fournit des passages documentaires, sans génération de réponse par LLM.

**Résultat :** fonctionnement du serveur MCP validé depuis
une application cliente externe.

**Suite :** documenter l'installation et faciliter la reproduction
de la démonstration sur un autre ordinateur.

### 19 octobre 2026 — intégration MCP et validations fonctionnelles

- **Branche :** `charles-bonus-mcp-integration`
- **Commit d'intégration :** `1319ab2`
- **Version de main intégrée :** `580edc8`

#### Intégration technique

- Fusion des travaux MCP avec la version de main contenant les contributions d'Alexis et Gaby.
- Conflits résolus dans `Docs/AVANCEMENT.md` et `uv.lock`.
- Dépendances MCP, agent, RPLS, DVF et Streamlit conservées.
- `uv lock --check` et `uv sync --locked` réussis.
- Tests automatisés : **186 réussis, 1 désélectionné**, en 29,34 secondes.
- Validation MCP réelle via un client `stdio` en sous-processus.
- Outil découvert : `search_docs`.
- Recherche DALO : 3 passages du document `F18005`, méthode `hybrid_rrf+rerank`, latence 8,3 s.
- Branche d'intégration publiée sur GitHub.
- Aucune modification apportée à `main`.

#### Vérifications fonctionnelles via Streamlit

- Interface Streamlit : lancement et affichage corrects.
- Baseline BM25 : recherche DALO fonctionnelle.
- Backend `rag` avec Ollama `qwen3:4b-instruct` : réponse DALO générée, citation `F18005`, latence 40,9 s.
- Backend `agent` : appel réussi à `check_eligibility` pour un couple avec un enfant à Lyon et 28 000 euros de revenu fiscal.
- Résultat éligibilité : PLAI non éligible ; PLUS et PLS éligibles.
- RPLS Créteil : **15 133 logements sociaux**, millésime 2025, confirmés par appel Python direct.
- DVF Créteil : 2024, 497 ventes et 3 548 euros/m² ; 2025, 608 ventes et 3 634 euros/m², confirmés par Python.
- Demande de logements disponibles aujourd'hui à Créteil : abstention correcte, aucun logement inventé.
- Backend `agent_tools` avec Ollama : recherche DALO, éligibilité et refus de divulgation du prompt système testés.
- Dans certains tests `agent_tools`, mécanisme de secours utilisé lorsque Qwen3 n'appelle pas `final_answer`.

#### Évaluation

- `eval/check_golden.py` : 52 questions structurellement valides (20 dev, 32 test).
- q036 (RPLS) et q037 (DVF) restent incomplètes et exclues des runs.
- Évaluation baseline BM25 : 50 questions exécutées, aucune erreur.
- Hit@5 baseline : 0,69 global ; 1,00 dev ; 0,35 test.
- Latence médiane baseline : 2 ms.
- Ces résultats valident l'exécution du pipeline, pas la justesse de toutes les réponses.

#### Anomalies restant à traiter

- Le backend `agent` associe parfois des citations juridiques non pertinentes aux résultats RPLS et DVF, malgré la provenance correcte renvoyée par les outils.
- Sur une injection de prompt, le backend `agent` ne divulgue pas ses instructions, mais produit une réponse hors sujet au lieu d'un refus explicite.
- Le backend `agent_tools` utilise parfois un repli de sortie structurée lorsque `final_answer` n'est pas appelé.
- Lors du test d'éligibilité, `agent_tools` appelle aussi `social_housing_stats`, outil non nécessaire à cette question.
- q036 demande le nombre de logements vacants, information absente du fichier RPLS utilisé : l'attente d'une réponse sans abstention doit être revue.
- q037 nécessite une réponse de référence calculée à partir des appartements vendus à Villeurbanne en 2025.
- Une évaluation end-to-end complète de la branche intégrée avec LLM reste à réaliser.

**Conclusion :** intégration MCP techniquement fonctionnelle et tests majeurs réussis sur les scénarios exécutés. Les anomalies de qualité doivent être signalées à l'équipe avant la livraison finale.

**Prochaines étapes :** finaliser `Docs/MCP.md`, transmettre les anomalies à Alexis, préparer une pull request pour revue et ne fusionner dans `main` qu'après validation de l'équipe.

### 20 octobre 2026 — Corrections et validation finale

**Branche :** `charles-finalisation-20261019`.

#### Correction des citations statistiques

- Modification de `src/agent/agent.py` sur une branche séparée.
- Ajout des citations officielles provenant des outils RPLS et DVF.
- Suppression des citations juridiques parasites dans les réponses statistiques lorsque la provenance de l outil est disponible.
- Deux tests de non-régression ajoutés dans `tests/test_agent_provenance.py`.
- Validation automatisée : 188 tests réussis, un désélectionné.
- Tests réels avec Ollama et Streamlit : RPLS Créteil et DVF Créteil validés.
- Les modifications du backend agent devront être relues par Alexis avant fusion.

#### Finalisation du golden set

- `q036` : abstention attendue pour la vacance des logements sociaux à Créteil, absente du fichier RPLS utilisé.
- `q037` : médiane DVF 2025 des appartements vendus à Villeurbanne : 3 567 EUR/m2, sur 1 890 ventes.
- `eval/check_golden.py` : 52 questions valides, sans avertissement.

#### Évaluations

- Baseline BM25 : 52 questions exécutées, aucune erreur.
- Baseline : Hit@5 = 0,69 ; MRR = 0,57.
- Évaluation Ollama `agent` sur trois questions : 3/3 traitées, sans erreur.
- Évaluation complète des 52 questions avec Ollama : en cours au moment de cette mise à jour.
- Les résultats définitifs seront ajoutés après la fin du run.

#### Travaux restants

- Analyser le rapport final des 52 questions avec Ollama.
- Examiner les échecs et les problèmes de routage ou de citation.
- Relancer les tests après toute correction supplémentaire.
- Préparer la revue par Alexis et Gaby.
- Publier les modifications, intégrer les branches validées et préparer le tag de livraison.


### 20 octobre 2026 — Bilan de fin de matinée

#### Évaluation complète de l'agent

- Backend : `agent`, modèle Ollama `qwen3:4b-instruct`.
- Golden set : 52 questions (20 développement, 32 test).
- Hit@5 : 0,80 ; MRR : 0,65.
- Précision des citations : 0,55.
- Routage correct : 0,96.
- Exactitude de l'éligibilité selon l'évaluation : 1,00.
- Une erreur d'exécution sur 52 questions : erreur HTTP 500 d'Ollama sur `q044`.
- Ces résultats correspondent au run initial, avant les derniers correctifs.
- Ils ne constituent pas une garantie d'exactitude juridique ou factuelle
  de toutes les réponses générées.

#### Correctifs de routage et de données DVF

- Routage ajusté : le transfert du bail après décès et le surloyer
  d'un locataire HLM relèvent de `documents`, et non d'`eligibilite`.
- Ajout des paramètres `type_local` et `annee` au routage DVF.
- Extraction déterministe d'une année explicitement mentionnée.
- En présence de plusieurs années distinctes, aucune année unique
  n'est choisie arbitrairement.
- Test réel avec Ollama pour `q037` : appel de `market_prices`
  avec Villeurbanne, Appartement, 2025.
- Résultat exact : 3 567 EUR/m2, 1 890 ventes, citation officielle DVF.
- Tests de non-régression ajoutés dans `tests/test_agent_dvf_filters.py`.

#### Analyse des limites du retrieval juridique

- `q021` : document pertinent `F1170`, présent dans le corpus,
  meilleur rang E5 21, meilleur rang RRF 45.
- `q022` : document pertinent `F21051`, présent dans le corpus,
  meilleur rang E5 24, meilleur rang RRF 59.
- Ces documents sont absents des 30 candidats transmis au reranker.
- L'expérience MiniLM à 60 candidats les classe respectivement
  aux rangs 35 et 44, donc toujours hors du top 5.
- Les réponses complètes de l'agent sur ces deux questions restent
  juridiquement insuffisantes malgré le routage corrigé.
- Configuration du retrieval conservée pour la livraison ;
  les variantes expérimentales ne sont pas adoptées.
- Les résultats expérimentaux sur ces questions de test ne doivent
  pas être présentés comme une évaluation indépendante.

#### Validation automatisée

- `uv run pytest -q` : 190 tests réussis, un désélectionné.
- `uv run python eval/check_golden.py` : 52 questions valides.
- `git diff --check` : aucune anomalie détectée.
- Branche de travail : `charles-finalisation-20261019`.
- Aucune fusion dans `main` à ce stade.

#### Prochaines actions

- Relire le diff final et vérifier la documentation.
- Créer un commit local de finalisation.
- Soumettre les changements à Alexis et Gaby pour revue.
- Ne fusionner dans `main` qu'après validation collective.

### 20 octobre 2026 — Validation fonctionnelle et revue GitHub

#### Démonstrations Streamlit

- Interface lancée avec `uv run streamlit run app/streamlit_app.py`.
- Backend `agent`, recherche hybride avec reranker,
  modèle Ollama `qwen3:4b-instruct`, 5 passages.
- DALO : réponse générée, document officiel `F18005`
  retrouvé et cité.
- Éligibilité : couple avec un enfant à Lyon, RFR de
  28 000 EUR ; outil `check_eligibility` exécuté.
  PLAI non éligible, PLUS et PLS éligibles.
- RPLS : Créteil, millésime 2025, 15 133 logements sociaux ;
  appel `social_housing_stats` réussi, source SDES citée.
- DVF : Villeurbanne, appartements vendus en 2025,
  prix médian de 3 567 EUR/m2 ; appel `market_prices`
  réussi, source DGFiP citée.
- Les quatre scénarios ont été validés dans l'interface.
- Certaines réponses prennent plusieurs dizaines de secondes
  avec le modèle Ollama exécuté localement.

#### Serveur MCP et dépendances

- `uv run pytest -q tests/test_mcp_server.py` :
  4 tests réussis en 6,65 secondes.
- Validation antérieure du protocole MCP depuis un client externe.
- `uv lock --check` : réussi, 125 packages résolus.
- `uv sync --locked` : réussi, 100 packages vérifiés.
- Version Python utilisée : 3.12.7.

#### GitHub et revue collective

- Branche : `charles-finalisation-20261019`.
- Dernier commit fonctionnel : `5bfd5c4`.
- Branche publiée et synchronisée avec GitHub.
- Pull request : `#4`, vers `main`.
- Adresse : https://github.com/Charles-Scannavino/Projet_Generative_AI/pull/4
- PR déclarée prête pour revue, sans conflit Git signalé.
- Demandes de revue envoyées à `Alexis5131` et `GabyD-web`.
- Aucune approbation reçue au moment de la vérification.
- Aucun contrôle CI GitHub rapporté sur la branche.
- Aucune fusion réalisée dans `main`.

#### Limites et travaux restants

- Réponses juridiques `q021` et `q022` encore insuffisantes ;
  limites du retrieval décrites dans la section précédente.
- Le succès des tests automatisés ne garantit pas
  l'exactitude juridique de toutes les réponses.
- Obtenir les revues d'Alexis et Gaby.
- Vérifier la version commune après éventuelle fusion.
- Préparer les scénarios et la présentation de soutenance.
- Ne pas fusionner sans validation collective.

### 20 octobre 2026 — Finalisation du bonus MCP complet

#### Fonctionnalités

- Serveur MCP `assistant-hlm`, SDK Python MCP 2.x,
  transport stdio.
- Quatre outils exposés et découverts :
  `search_docs`, `check_eligibility`,
  `social_housing_stats`, `market_prices`.
- Réutilisation directe du moteur RAG et des fonctions métier.
- Entrées validées par Pydantic ; résultats structurés.
- Conservation des sources officielles, des années de données
  et des avertissements métier.
- Aucun calcul réglementaire effectué par un LLM.

#### Tests automatisés

- Neuf tests MCP réussis.
- Tests d'exécution des trois nouveaux outils avec données réelles.
- Cinq scénarios d'erreur métier testés.
- Suite complète : 195 tests réussis, un désélectionné.
- Golden set : 52 questions valides (20 dev, 32 test).
- `git diff --check` : aucune anomalie.

#### Démonstration MCP Inspector v2.11.0

- Connexion STDIO : réussie.
- Version de protocole affichée : MCP 2025-11-25.
- Découverte des quatre outils : réussie.
- search_docs : DALO, document F18005 au premier rang,
  méthode hybrid_rrf+rerank ; appel environ 9,39 s.
- check_eligibility : Lyon, foyer de trois personnes,
  RFR 28 000 EUR ; PLUS et PLS éligibles, PLAI non ;
  appel environ 41 ms.
- social_housing_stats : Créteil, 15 133 logements
  au 1er janvier 2025 ; appel environ 41 ms.
- market_prices : Villeurbanne, Appartement, 2025,
  1 890 ventes, médiane 3 567 EUR/m2 ;
  appel environ 1,86 s.

Ces latences sont des observations ponctuelles, pas des
moyennes ni des benchmarks comparatifs.

#### Documentation et limites

- README : procédure d'installation et d'utilisation du MCP.
- Docs/MCP.md : contrats des outils, exemples et dépannage.
- Connexion à Claude Desktop proposée comme prolongement ;
  aucune validation avec un agent IA externe à ce stade.
- MCP Inspector ne produit pas de réponse par un LLM.
- RPLS ne renseigne pas la disponibilité en temps réel.
- DVF ne renseigne pas les loyers.
- Les limites du retrieval juridique q021/q022 subsistent.

#### Livraison

- Version initiale fusionnée dans main via la PR #4,
  commit 8c0e567.
- Extension développée sur charles-bonus-mcp-complet.
- Statut final : extension MCP fusionnée via la PR #5 (commit `32ec0a3`) ; le tag `v1.0` pointe vers le commit ultérieur `24702f` d'ablation retrieval sur le split test.

### 20 octobre 2026 — Bilan final et soutenance

Cette entrée met à jour la situation du projet à la livraison. Les descriptions d'anciennes étapes encore visibles plus haut constituent un **historique daté**, et non une liste actualisée des travaux à venir.

#### Livrable validé
- Corpus : **455 documents**, **1 357 passages**, chunking `fixed_300_60`.
- Retrieval hybride BM25/E5/RRF/MiniLM, RAG, agent et trois outils métier fonctionnels.
- Interface Streamlit fonctionnelle, avec sources et trace d'exécution.
- Serveur MCP proposant quatre outils, vérifiés dans MCP Inspector v2.11.0 ; **9 tests MCP réussis**.
- Validation finale sur `main` : **195 tests réussis**, un désélectionné ; **52 questions structurellement valides** (20 développement, 32 test).
- PR #4 et #5 fusionnées ; bonus MCP au commit `32ec0a3` ; tag **`v1.0`** actuellement associé au commit `24702f`. Les corrections documentaires sur `main` n'affectent pas ce tag.

#### Démonstration de soutenance
- **Un enregistrement vidéo Streamlit de secours a été réalisé** par le présentateur ; son emplacement de stockage n'est pas documenté dans le dépôt.
- **Une démonstration Streamlit en direct est prévue le 21 octobre 2026**, avec repli sur la vidéo en cas d'imprévu.
- Scénarios préparés : DALO (fiche `F18005`), éligibilité à Lyon, RPLS à Créteil et DVF à Villeurbanne.
- MCP Inspector permet de montrer les quatre outils MCP sans avoir besoin du LLM local.

#### Limites et tâches non closes
- L'évaluation exhaustive de `agent_tools --provider ollama` n'est pas documentée comme terminée ; ne pas la confondre avec le run local `agent`.
- Aucune comparaison finale documentée contre une baseline DINUM.
- Contrôle automatisé Gitleaks effectué sur 128 commits (370,97 Mo) ; aucune conclusion exhaustive sur l'absence de secrets.
- Variante hiérarchique de découpage non évaluée séparément.
- Des erreurs documentaires `q021`/`q022` restent connues ; les 52 questions valides ne sont pas 52 réponses toutes correctes.
- Connexion à Claude Desktop non testée.

### 20 octobre 2026 — Contrôle de sécurité Git (Gitleaks)

- Vérification de la liste des fichiers suivis : seul `.env.example` est apparu parmi les noms recherchés ; `.env` est ignoré selon `git check-ignore -v .env` (`.gitignore`, ligne 227).
- Gitleaks **8.30.1** exécuté en lecture seule sur l'historique accessible localement, avec `--log-opts="--all"` et masquage des valeurs (`--redact=100`).
- **128 commits** analysés, **370,97 Mo** parcourus.
- Le contrôle automatisé a été exécuté et documenté. Il ne constitue pas une certification d'absence de secrets ; une vérification manuelle resterait nécessaire pour lever toute incertitude.
- Aucune modification de l'historique Git ni du tag `v1.0` n'a été réalisée. Le rapport local n'a pas été publié sur GitHub.

### 20 octobre 2026 — RAG de Charles intégré et fournisseur Anthropic (Alexis)

#### RAG (`src/agent/rag.py`)

- `answer_with_rag` reprend la logique de la PR #2 (Charles) : prompt système
  à 7 règles, sortie JSON contrainte `{answer, citation_document_ids, abstained}`,
  passages transmis au LLM en JSON (document_id, source, titre, article, URL, texte).
- Adaptation au contrat commun : `AssistantResponse`, façade `ask()` et
  fonctions partagées (`passages_from`, `unsupported_numbers`, `error_response`)
  conservées ; aucune exception ne remonte à l'interface ni à `run_eval`.
- Contrôles stricts de Charles convertis en abstention tracée dans `warnings` :
  citation hors contexte, réponse non abstenue sans citation ; une abstention
  ne garde aucune citation. Les citations LEGI portent leur numéro d'article.
- Fournisseur au choix : Ollama (défaut), OpenRouter ou Anthropic.

#### Fournisseur Anthropic (bonus local vs API)

- `AnthropicClient` (SDK officiel `anthropic` 1.16.0), clé `ANTHROPIC_API_KEY`
  lue uniquement depuis `.env`, modèle par défaut **Claude Opus 5.5**
  (`claude-opus-5-5`, effort `medium`, `max_tokens` 16 000, 4 $ / 20 $ par MTok).
- Disponible pour `rag`, `agent_tools` et le juge de `run_eval`
  (`--provider anthropic`, `--judge-provider anthropic`), ainsi que dans l'UI.
- Contraintes d'Opus 5.5 respectées : ni `temperature` ni `seed`,
  `tool_choice` automatique, blocs de raisonnement renvoyés tels quels,
  refus (`stop_reason = refusal`) converti en réponse `error`.
- Repli serveur `fallbacks: "default"` activé (modèle réellement utilisé
  enregistré dans `model`).

#### Vérifications

- Suite complète : 215 tests réussis (20 nouveaux : `tests/test_rag.py`,
  `tests/test_anthropic.py` avec SDK simulé), 2 désélectionnés (intégration).
- `rag` réel via OpenRouter (`openai/gpt-4.1-mini`) : DALO cité F18005,
  4,6 s, 0,0014 $ ; disponibilité temps réel à Lyon : abstention sans citation.
- `rag` réel via Ollama (`qwen3:4b-instruct`, MacBook Air M2 8 Go) : DALO
  cité F18005, 63 s chargement du modèle compris.
- Anthropic non testé en réel : `ANTHROPIC_API_KEY` absente du `.env`.
  Test d'intégration prêt : `uv run pytest -m integration tests/test_anthropic.py`.

#### Reste à faire

- Renseigner `ANTHROPIC_API_KEY`, lancer le test d'intégration puis un run
  `agent_tools --provider anthropic` pour compléter la comparaison local vs API.
- Fermer la PR #2, désormais intégrée.

### 21 octobre 2026 — Validation d'intégration sans notebook

- Branche d'essai `test-alexis-sans-notebook` : retrait de `notebooks/Assistant_HLM_RAG.ipynb` validé par Alexis.
- Suite Python `uv run pytest -q` : **215 tests réussis, 2 désélectionnés** après retrait du notebook.
- Suite MCP `uv run pytest -q tests/test_mcp_server.py` : **9 tests réussis**.
- Retrieval hybride vérifié par la question DALO : document **F18005** retrouvé au premier rang.
- RAG local `--backend rag --provider ollama` : réponse sur le DALO avec citation Service-Public F18005.
- Streamlit testé en réel avec le RAG, puis avec `agent_tools --provider ollama` : appels fonctionnels à `check_eligibility` et `social_housing_stats`.
- Les corrections RAG d'Alexis et la suppression du notebook ne nécessitent aucun fichier `.ipynb` pour les chemins d'exécution testés.
- Anthropic : tests avec SDK simulé couverts par la suite automatique ; appels réels non validés faute de `ANTHROPIC_API_KEY` dans le `.env` local.
- Le tag historique `v1.0` reste inchangé. Les contrôles automatisés et démonstrations réalisés ne prouvent pas l'exécution exhaustive de tous les scénarios en production.

