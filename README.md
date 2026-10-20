# Projet Generative AI — Assistant HLM RAG & Agent

Projet de groupe du cours **Generative AI I** : construction d'un assistant local sur le logement social français combinant **RAG**, **recherche hybride**, **reranking**, **LLM local avec Ollama** et **outils déterministes** sur données ouvertes.

## Objectif

L'assistant doit pouvoir :

- expliquer les droits et démarches liés au logement social ;
- répondre à partir de sources officielles Service-Public et LEGI ;
- vérifier l'éligibilité à partir de barèmes calculés en Python ;
- interroger des statistiques communales RPLS ;
- calculer des indicateurs DVF sur les prix de vente ;
- citer les sources utilisées ;
- s'abstenir lorsque les données disponibles ne permettent pas de répondre de façon fiable.

L'objectif n'est pas seulement d'obtenir une démonstration fonctionnelle : chaque composant important doit être **mesuré** sur un golden set avec qualité, latence, tokens, appels d'outils et coût.

## Architecture générale

Le pipeline documentaire cible est :

```text
Documents Service-Public + LEGI
        ↓
Nettoyage / normalisation
        ↓
Chunking + métadonnées
        ↓
BM25 ─────────────┐
                  ├── RRF ──> Reranker ──> Top-k chunks
Dense E5 ─────────┘
                                ↓
                         Ollama / Qwen3
                                ↓
                   Réponse structurée + citations
```

L'architecture agentique cible est :

```text
Utilisateur
    ↓
Agent / LLM local
    ├── search_docs
    ├── check_eligibility
    ├── social_housing_stats
    └── market_prices
    ↓
Réponse structurée + sources + trace des outils
```

Les calculs réglementaires et statistiques restent déterministes en Python : le LLM orchestre les outils mais ne remplace pas les calculs.

## État actuel

### Corpus

Le corpus documentaire RAG est finalisé :

- **117 documents Service-Public** ;
- **338 articles LEGI** ;
- **455 documents** au total ;
- chunking de référence : **`fixed_300_60`** ;
- **1 357 chunks** dans le corpus de retrieval.

Fichier de chunks actuellement utilisé :

```text
data/processed/chunks/fixed_300_60.jsonl
```

### Retrieval

Le retrieval final est exposé par une seule fonction, `search_docs`, utilisée par l'agent et par l'évaluation. Pipeline, sur les 1 357 chunks de `fixed_300_60` :

```text
Question
   ├── BM25  (rank_bm25.BM25Okapi)               → top 100 chunks
   └── Dense (intfloat/multilingual-e5-small)    → top 100 chunks
            ↓ fusion RRF : score = Σ 1 / (60 + rang)
   Reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 sur les 30 premiers candidats
            ↓
   k passages + trace (rangs BM25 / dense, scores RRF / rerank)
```

Modèles (noms exacts Hugging Face) :

| Rôle | Modèle | Taille | Remarque |
|---|---|---:|---|
| Lexical | `rank_bm25.BM25Okapi` | — | paramètres par défaut (k1 = 1.5, b = 0.75) |
| Dense | `intfloat/multilingual-e5-small` | 471 Mo | 384 dimensions, préfixes `query:` / `passage:` |
| Reranker retenu | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 471 Mo | 118 M paramètres |
| Reranker comparé | `BAAI/bge-reranker-v2-m3` | 2 271 Mo | 568 M paramètres, ablation uniquement |

#### Ablation

Évaluation sur les **18 questions RAG** du golden set. Les métriques sont calculées au niveau **document** (le meilleur chunk d'un document donne son rang). Latence par requête mesurée sur MacBook Pro M1 (MPS), après une requête d'échauffement. « Chargement » = construction ou chargement des composants nécessaires, embeddings denses lus depuis le cache (leur calcul, sans cache, prend 17,9 s).

| Méthode | R@1 | R@5 | R@10 | MRR | Latence moy. | p95 | Chargement |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6944 | 1.0000 | 1.0000 | 0.8306 | 2.3 ms | 3.1 ms | 0.10 s |
| Dense E5 | 0.5833 | 0.9444 | 1.0000 | 0.7394 | 9.5 ms | 12.9 ms | 3.94 s |
| Hybride RRF | 0.7500 | 0.9444 | 1.0000 | 0.8524 | 10.5 ms | 11.3 ms | 4.04 s |
| **Hybride + MiniLM** (défaut) | **0.8611** | **1.0000** | **1.0000** | **0.9444** | 452.4 ms | 502.7 ms | 6.74 s |
| Hybride + bge | 0.9167 | 1.0000 | 1.0000 | 0.9722 | 4281.7 ms | 4427.8 ms | 8.99 s |

Lecture :

- l'hybride RRF améliore le MRR de BM25 (0.8306 → 0.8524), mais fait sortir **q006 / F21051** du top 5 (rang 7, comme le dense seul) ;
- les deux rerankers remettent q006 au rang 1 ; il reste au rang 2 q005 (F869) et q013 (F31864) avec MiniLM, et q013 seulement avec bge ;
- 9 questions sur 18 ont un rang différent selon la méthode ; le détail est dans `eval/retrieval/details.csv`.

#### Configuration retenue

La règle de choix a été fixée **avant** l'expérience : prendre le reranker au meilleur MRR, sauf si l'écart est inférieur à 0,02 (bruit sur 18 questions) ou si la latence p95 du plus lourd dépasse 2 s sur ce Mac ; dans ces deux cas, prendre le plus léger. bge est meilleur de +0.0278 en MRR, mais sa p95 vaut 4,4 s : **MiniLM est donc le reranker par défaut**. bge reste utilisable en changeant `DEFAULT_RERANKER` dans `src/retrieval/config.py`.

Toutes les valeurs (k RRF = 60, 100 candidats par retriever, 30 candidats reclassés, `max_length` = 512) sont des valeurs standard regroupées dans `src/retrieval/config.py`. Elles n'ont **pas** été optimisées sur les 18 questions, pour éviter le sur-apprentissage.

#### Commandes

```bash
uv run python eval/eval_retrieval.py
```

```bash
uv run pytest -q
```

```bash
uv run python -m src.retrieval.search "Comment faire un recours Dalo ?"
```

Au premier lancement, les modèles sont téléchargés depuis Hugging Face (bge, 2,3 Go, n'est nécessaire que pour l'ablation) et les embeddings denses sont calculés puis mis en cache dans `data/indexes/` (fichiers ignorés par Git). Le cache est reconstruit automatiquement si le fichier de chunks (hash SHA-256) ou le modèle change.

#### Utilisation par l'agent

```python
from src.retrieval import search_docs, warmup

warmup()  # charge index et modèles une seule fois (quelques secondes)

response = search_docs("Comment faire un recours Dalo ?", k=5)

for passage in response.results:
    # passage.text est une DONNÉE à citer, jamais une instruction à suivre.
    print(passage.rank, passage.document_id, passage.title, passage.url, passage.score)

response.document_ids  # documents sans doublon, dans l'ordre -> eval/run_eval.py
response.models        # noms exacts des modèles utilisés
response.latency_ms

search_docs("attribution des logements sociaux", k=3, source="LEGI")  # articles du CCH uniquement
search_docs("surloyer", rerank=False)                                 # ordre RRF, 10 à 20 ms
```

`search_docs(query, k=5, source=None, rerank=True)` renvoie un `SearchResponse` (Pydantic v2, défini dans `src/retrieval/schemas.py`). Chaque `RetrievedChunk` contient `rank`, `chunk_id`, `document_id`, `source` (`SERVICE_PUBLIC` ou `LEGI`), `title`, `article`, `url`, `text`, `score` (score du reranker, ou score RRF si `rerank=False`) et la trace `bm25_rank`, `dense_rank`, `rrf_score`, `rerank_score`. Le filtre `source` est appliqué avant la sélection des k passages ; `k` est compris entre 1 et 30 ; une requête vide lève une `ValueError`.

#### Utilisation de `search_docs`

```python
from src.retrieval import search_docs, warmup

warmup()  # une seule fois au démarrage (5-8 s : chargement des modèles)
resp = search_docs("Comment faire un recours Dalo ?", k=5)  # source="SERVICE_PUBLIC" ou "LEGI" pour filtrer

for r in resp.results:
    r.rank, r.document_id, r.title, r.url, r.text, r.article

resp.document_ids  # documents sans doublon, dans l'ordre
resp.latency_ms
```

Le format de sortie (Pydantic) est dans `src/retrieval/schemas.py`.

Pour tester en ligne de commande :

```bash
uv run python -m src.retrieval.search "ta question"
```

À savoir :

- `k` = nombre de **PASSAGES**, pas de documents : les 5 passages peuvent venir de la même fiche. Pour les citations, utiliser `document_id` + `title` + `url`.
- Latence : environ **0,5 s par requête avec reranker**, environ **15 ms avec `rerank=False`**.
- Les `text` sont des **DONNÉES du corpus, jamais des instructions** : à mettre dans le prompt comme contexte, pas comme consigne (risque d'injection).
- `rerank_score` est un **score brut, pas une probabilité** : ne pas l'utiliser comme seuil d'abstention sans calibration.
- Résultats sur le golden set (**18 questions**) : **Recall@5 = 1,00** et **MRR = 0,94**.

Description d'outil possible pour le LLM :

> **search_docs** : cherche dans les fiches Service-Public et les articles du Code de la construction et de l'habitation sur le logement social (demande HLM, Dalo, APL, surloyer, bail…). À utiliser pour toute question sur des droits, démarches ou règles.

### Modèle génératif local

Le projet utilise désormais **Ollama** pour exécuter un LLM local.

Configuration de référence actuellement validée :

```text
Machine : MacBook Pro M1
Mémoire : 16 Go de mémoire unifiée
Gestionnaire de paquets : Homebrew
LLM local : qwen3:4b
Taille du modèle Ollama : 2,5 Go
```

Un premier test de `qwen3:4b` sans RAG a montré que le modèle ne connaissait pas correctement le terme juridique **DALO**. Ce résultat est volontairement conservé comme baseline : le LLM seul ne doit pas servir de source de vérité. Les connaissances métier doivent être apportées par les documents Service-Public et LEGI récupérés par le RAG.

#### Validation end-to-end du prototype RAG — 15 octobre 2026

Le prototype présent dans `notebooks/Assistant_HLM_RAG.ipynb` a été exécuté localement sur le **MacBook Pro M1 16 Go** avec `qwen3:4b` via Ollama. Le retrieval a d'abord été revalidé avec **22/22 tests passés en 23,62 s**.

| Test | Résultat | Sources / décision | Latence totale observée |
|---|---|---|---:|
| « Comment déposer une demande de logement social ? » | réponse correcte, `abstained=false` | `F10007` + `F869` | 137,9 s au premier appel ; 122,8 s à chaud |
| « Qu'est-ce que le DALO ? » | **« Droit au logement opposable »**, `abstained=false` | `F18005` | 93,2 s |
| « Quels logements HLM sont disponibles aujourd'hui à Créteil ? » | `abstained=true`, aucune citation fabriquée | retrieval `F10007`, mais source insuffisante pour une disponibilité temps réel | 67,4 s |

Le cas DALO constitue une démonstration utile : **le LLM seul se trompait**, alors que le RAG retrouve `F18005` et corrige la réponse. Les identifiants de citations proposés par le modèle sont validés contre les documents effectivement récupérés.

La chaîne end-to-end **retrieval → Ollama/Qwen3 → sortie Pydantic → citations/abstention** est donc fonctionnelle. La latence générative reste toutefois trop élevée pour une démonstration fluide et doit être optimisée/mesurée plus finement dans l'évaluation finale.

### Outils déterministes

Les calculs ne sont jamais confiés au LLM : trois outils Python (`src/tools/`) renvoient des sorties Pydantic avec `provenance` (source, millésime) et `caveats` (limites des données).

| Outil | Données | Ce qu'il renvoie | Hors périmètre |
|---|---|---|---|
| `check_eligibility` | Plafonds PLAI / PLUS / PLS **2026** (arrêté du 19/12/2025), saisis dans `data/processed/plafonds_hlm_2026.csv` | Catégorie de ménage, zone, plafond et écart pour chaque financement | DOM (régime distinct) |
| `social_housing_stats` | RPLS au **1er janvier 2025** (SDES), agrégé par commune | Nombre de logements sociaux, financement initial, taille, QPV, ancienneté, DPE | Disponibilité, vacance, loyers |
| `market_prices` | DVF **2024 et 2025**, France entière, agrégé par commune | Prix médian au m², quartiles, nombre de ventes | Loyers ; Alsace-Moselle et Mayotte |

Les communes sont résolues dans le COG Insee 2026 (`src/tools/communes.py`) : homonymes signalés avec leurs candidats, arrondissements de Paris/Lyon/Marseille, communes déléguées rattachées à leur commune nouvelle.

Reconstruire les données (les fichiers bruts ne sont pas versionnés, les agrégats le sont) :

```bash
uv run python -m src.preprocessing.build_communes_ref   # télécharge le COG
uv run python -m src.ingestion.rpls && uv run python -m src.preprocessing.build_rpls_stats
uv run python -m src.ingestion.dvf  && uv run python -m src.preprocessing.build_dvf_stats
```

### Backends de l'assistant

Tous les backends renvoient le même contrat Pydantic `AssistantResponse` (`src/agent/schemas.py`), utilisé par l'interface et par `eval/run_eval.py`. Point d'entrée unique : `from src.agent import ask`.

| Backend | Principe | Modèle |
|---|---|---|
| `baseline` | BM25 seul, sans LLM (`src/agent/baseline.py`) | aucun |
| `rag` | `search_docs` puis synthèse citée (`src/agent/rag.py`) | Ollama `qwen3:4b-instruct` |
| `agent` | recherche systématique + formulaire JSON rempli par le LLM ; Python choisit et exécute l'outil (`src/agent/agent.py`), conçu pour un petit modèle local | Ollama `qwen3:4b-instruct` |
| `agent_tools` | **tool calling natif** : le LLM enchaîne `search_docs` et les trois outils, puis répond via un pseudo-outil `final_answer` (`src/agent/tool_agent.py`) | `openai/gpt-4.1-mini` via OpenRouter (`temperature = 0`, `seed = 42`), ou Ollama |

Configuration de `agent_tools` : `configs/agent.yaml` et `configs/tools.yaml` (lus par `src/settings.py`), prompts dans `prompts/*.md.j2` (Jinja2). Le secret `OPENROUTER_API_KEY` est lu uniquement depuis `.env` (voir `.env.example`). Toute valeur YAML peut être surchargée par variable d'environnement, ex. `HLM_AGENT__BACKEND=ollama`.

```bash
cp .env.example .env    # puis renseigner OPENROUTER_API_KEY
uv run python -m src.agent "Avec 28 000 € pour un couple avec un enfant à Lyon, suis-je éligible ?"
uv run python -m src.agent "Combien de logements sociaux à Créteil ?" --backend agent_tools --provider ollama --json
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --backend agent_tools --run-name agent_tools_gpt41mini
```

**Garde-fous contre la prompt injection** (`src/agent/guards.py`, `src/agent/tools_registry.py`) :

- entrées d'outils validées strictement (champs inconnus refusés, pas de coercition, noms de commune contraints puis résolus dans une liste fermée) ; le LLM recopie des faits (en couple, âges, personnes à charge), Python calcule la catégorie de ménage et le « jeune ménage » ;
- registre d'outils fermé, quota d'appels par outil, arrêt sur appels répétés ; `search_docs` n'expose au modèle que la requête (ni `k` ni filtre de source) ;
- sorties renvoyées au LLM dans une enveloppe JSON `untrusted_data`, marqueurs de rôle (`<|im_start|>`, `[INST]`…) neutralisés, passages tronqués ;
- une commune absente de la question est refusée ; un département « deviné » pour lever une homonymie est retiré ;
- citations limitées aux documents réellement récupérés (titre et URL tirés du corpus) ; citations retirées et chiffres absents des sources signalés dans `warnings` (métriques `invented_citation` et `unsupported_number` de `run_eval`).

Résultats du 18 octobre 2026 avec `run_eval.py` (golden v2, 50 questions évaluables, juge `google/gemini-2.5-flash` via OpenRouter, d'une autre famille que les modèles évalués) :

| Run | Modèle | Hit@5 dev / test | Justesse (juge) | Abstention correcte | Routage | Éligibilité exacte | Citations inventées | Chiffres non ancrés | Latence p50 / p95 | Coût agent |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_bm25` | aucun (extractif) | 1,00 / 0,35 | – | 0,84 | 0,86 | 0,00 | 0 | – | 0,02 / 0,04 s | 0 |
| `agent_tools_gpt41mini` | `openai/gpt-4.1-mini` | 0,94 / 0,59 | **0,89** | 0,94 | 0,94 | 1,00 | 0 | 0,02 | 7,5 / 14,4 s* | 0,086 $ |
| `agent_tools_mistral_small_24b` | `mistralai/mistral-small-3.2-24b-instruct` | 0,89 / 0,59 | 0,85 | 0,94 | 0,98 | 1,00 | 0 | 0,04 | 5,8 / 12,7 s | **0,037 $** |
| `agent_tools_qwen3_30b_a3b` | `qwen/qwen3-30b-a3b-instruct-2507` | 0,89 / 0,65 | 0,61 | 0,66 | 0,90 | 1,00 | 0 | 0,02 | 25,6 / 46,1 s | 0,089 $ |

\* Machine chargée pendant ce run ; sans charge (17 octobre) : p50 4,1 s, p95 4,9 s. Coût du juge : environ 0,014 $ par run.

Lecture : `gpt-4.1-mini` est le meilleur compromis (backend par défaut). Mistral Small est 2,3 fois moins cher, mais a recopié des plafonds faux dans 2 réponses d'éligibilité sur 5 (signalés par `unsupported_number`, la réponse reste fausse). Qwen3-30B enchaîne mal les outils (46 appels répétés, 32 communes inventées refusées) et s'abstient à tort sur 12 questions RAG.

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ --backend agent_tools \
    --model openai/gpt-4.1-mini --judge --judge-provider openrouter --judge-model google/gemini-2.5-flash \
    --run-name agent_tools_gpt41mini
uv run python eval/run_eval.py --compare      # tableau de tous les runs (eval/runs/comparison.md)
```

**Mode local (Ollama)** : `agent_tools --provider ollama` passe par l'API native `/api/chat` (client partagé `src/agent/ollama.py`, contexte `OLLAMA_NUM_CTX` = 8 192 à chaque requête). L'endpoint compatible OpenAI `/v1` ignore la taille de contexte demandée : avec le contexte par défaut du serveur (4 096 tokens sur un Mac à 8 Go), le 2e tour de l'agent (≈ 5 200 tokens) était refusé. Quand `qwen3:4b-instruct` répond en texte libre au lieu d'appeler `final_answer`, l'agent fait reformuler la réponse en sortie JSON contrainte (citations et abstention récupérées). Vérifié le 18 octobre sur un MacBook Air M2 8 Go (Ollama 0.40.6) : Dalo (citation F18005), éligibilité, DVF, demande de précision et injection corrects avec `agent_tools`, ainsi qu'avec les backends `agent` et `rag` ; 9 à 75 s par question. Sur une machine à 8 Go, ne lancer qu'un seul processus (évaluation ou UI) à la fois avec Ollama.

Détails et analyse d'erreurs dans [Docs/AVANCEMENT.md](Docs/AVANCEMENT.md) et `eval/runs/`.

## Installation locale

### 1. Prérequis

Environnement de référence :

- macOS sur Apple Silicon ;
- Homebrew ;
- Python 3.12+ ;
- `uv` ;
- Ollama.

Vérifier Homebrew :

```bash
brew --version
```

Installer `uv` si nécessaire :

```bash
brew install uv
```

Installer Ollama si nécessaire :

```bash
brew install ollama
```

### 2. Cloner le repository

Le notebook et plusieurs scripts supposent actuellement que le projet se trouve dans :

```text
~/Documents/Projets/Projet_Generative_AI
```

Préparer l'emplacement puis cloner le dépôt :

```bash
mkdir -p ~/Documents/Projets
cd ~/Documents/Projets
git clone https://github.com/Charles-Scannavino/Projet_Generative_AI.git
cd Projet_Generative_AI
```

Le repository est privé : le compte GitHub utilisé doit avoir accès au projet.

### 3. Installer l'environnement Python

Le projet est géré avec `uv`. Les dépendances sont définies dans `pyproject.toml` et verrouillées dans `uv.lock`.

Depuis la racine du projet :

```bash
uv sync
```

Vérifier Python :

```bash
uv run python --version
```

L'environnement actuellement validé utilise Python **3.12.7**.

### 4. Démarrer Ollama

Si Ollama a été installé avec Homebrew, il peut être lancé comme service :

```bash
brew services start ollama
```

Vérifier qu'Ollama répond :

```bash
ollama --version
```

### 5. Télécharger le modèle local

Le modèle retenu est la variante **instruct** de Qwen3 4B :

```bash
ollama pull qwen3:4b-instruct
```

Attention : le tag `qwen3:4b` pointe vers **Qwen3-4B-Thinking-2507**, qui raisonne toujours avant de répondre, même avec `think=false` (vérifié sur la page des tags Ollama et en local). C'est la cause probable des 67 à 138 s par question mesurées au départ. `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507) a la même taille sans cette phase de raisonnement.

Vérifier les modèles installés :

```bash
ollama list
```

La configuration validée affiche actuellement :

```text
qwen3:4b-instruct    2.5 GB
```

### 6. Tester le modèle

Lancer une session interactive :

```bash
ollama run qwen3:4b-instruct
```

Exemple :

```text
Explique en français ce qu'est le DALO en trois phrases.
```

Pour quitter :

```text
/bye
```

Qwen3 peut afficher un mode de raisonnement `Thinking...`. Pour les tests documentaires simples, on peut le désactiver dans la session interactive :

```text
/set nothink
```

Le mode de raisonnement devra être évalué séparément si nous décidons de l'utiliser, car il augmente le nombre de tokens et la latence.

## Installation reproductible depuis GitHub

Cette procédure permet de préparer le projet sur un nouvel ordinateur,
sans dépendre des chemins locaux utilisés pendant le développement.

### Étape 1 — Cloner le dépôt

Le dépôt GitHub est privé : disposer des autorisations nécessaires.

~~~bash
git clone https://github.com/Charles-Scannavino/Projet_Generative_AI.git
cd Projet_Generative_AI
~~~

La version complète intégrant le MCP et les dernières corrections
sera disponible sur main après validation et fusion des branches.

### Étape 2 — Installer Python et les dépendances

Prérequis : Python 3.12, uv et Git.

Sur macOS avec Homebrew :

~~~bash
brew install uv python@3.12
~~~

Depuis la racine du projet :

~~~bash
uv sync --python 3.12 --locked
uv run python --version
~~~

La commande uv sync installe les dépendances définies dans
pyproject.toml et verrouillées dans uv.lock.

Sur un nouvel ordinateur, uv peut créer un environnement .venv
à la racine du dépôt.

### Étape 3 — Installer et démarrer Ollama

Sur macOS avec Homebrew :

~~~bash
brew install ollama
brew services start ollama
~~~

Télécharger le modèle génératif retenu :

~~~bash
ollama pull qwen3:4b-instruct
ollama list
~~~

Modèle utilisé : qwen3:4b-instruct, environ 2,5 Go.

Le modèle fonctionne localement : aucune clé API OpenRouter
est nécessaire pour les backends Ollama.

### Étape 4 — Télécharger les modèles de recherche documentaire

Le retrieval utilise deux modèles Hugging Face :

- Embeddings : intfloat/multilingual-e5-small
- Reranker : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

Ces modèles sont chargés automatiquement par sentence-transformers
lors de la première utilisation du moteur de recherche.

Depuis la racine du projet, lancer :

~~~bash
uv run python -c "from src.retrieval import warmup; warmup(); print(\"Retrieval pret\")"
~~~

Au premier lancement, les poids des modèles peuvent être téléchargés
depuis Hugging Face. Une connexion Internet est alors nécessaire.

Les modèles téléchargés sont conservés dans les caches locaux
Hugging Face. Les embeddings et index calculés par le projet
sont mis en cache dans data/indexes/.

Les lancements suivants peuvent réutiliser les modèles et caches
existants, si leurs versions et les données source ne changent pas.

Le modèle BAAI/bge-reranker-v2-m3, plus volumineux, ne fait pas
partie du pipeline par défaut. Il sert aux comparaisons expérimentales.

### Étape 5 — Vérifier les données du projet

Le retrieval nécessite notamment :

- data/processed/rag_documents.jsonl
- data/processed/chunks/fixed_300_60.jsonl

Les outils statistiques utilisent aussi les fichiers traités RPLS,
DVF et les référentiels de communes.

Vérifier les fichiers documentaires :

~~~bash
ls -lh data/processed/rag_documents.jsonl
ls -lh data/processed/chunks/fixed_300_60.jsonl
~~~

Les principaux fichiers de données traitées nécessaires au
fonctionnement du projet sont versionnés dans GitHub :

- rag_documents.jsonl et les chunks de référence ;
- communes_ref.csv ;
- rpls_communes.csv ;
- dvf_communes.csv ;
- plafonds_hlm_2026.csv.

Ils sont donc récupérés automatiquement lors du clonage.

Les données brutes supplémentaires, les modèles téléchargés et les
caches restent distincts et ne sont pas nécessairement versionnés.

Si un fichier nécessaire manque après le clonage, il faut récupérer
les données prévues par le projet et suivre les étapes de préparation
décrites dans le README et dans les scripts de preprocessing.

Installer les dépendances et télécharger les modèles ne suffit pas
à reconstituer automatiquement un jeu de données absent.

### Étape 6 — Tester la recherche documentaire

~~~bash
uv run python -m src.retrieval.search "Qu est-ce que le DALO ?"
~~~

Résultat attendu : une recherche documentaire contenant notamment
la fiche Service-Public F18005, avec les passages et leurs sources.

### Étape 7 — Tester les composants

~~~bash
uv run python eval/check_golden.py
uv run pytest -q
~~~

Ces commandes vérifient le jeu de référence et les tests automatisés
exécutables sans appel réel à une API LLM.

### Étape 8 — Lancer la plateforme Streamlit

Depuis la racine du projet :

~~~bash
uv run streamlit run app/streamlit_app.py
~~~

Ouvrir dans le navigateur le lien affiché par Streamlit,
généralement http://localhost:8501.

Configuration locale recommandée :

- Fournisseur : Ollama.
- Modèle : qwen3:4b-instruct.
- Retrieval : hybride avec reranker.
- Nombre de passages : 5.

Les backends baseline, rag, agent et agent_tools permettent
de comparer plusieurs stratégies de réponse.

### Étape 9 — Utiliser le bonus MCP

Le serveur MCP peut être démarré avec :

~~~bash
uv run python -m src.mcp_server
~~~

Il utilise le transport stdio et attend un client MCP.
Il ne présente donc pas directement une interface dans le Terminal.

Consulter Docs/MCP.md pour la documentation complète,
les tests et la démonstration avec MCP Inspector.

### Étape 10 — Reproductibilité et limites

- Python et les bibliothèques sont gérés par uv.
- Le modèle génératif est téléchargé via Ollama.
- E5 et MiniLM sont récupérés via Hugging Face.
- Les index et embeddings peuvent être reconstruits automatiquement
  lorsque les données traitées nécessaires sont disponibles.
- Les temps de chargement dépendent de la machine et des caches.
- Une installation neuve doit être validée sur les fichiers réellement
  distribués avec la version finale du projet.

---

## Ollama : rôle dans le projet

Ollama **n'est pas un moteur de recherche Web**. Dans notre configuration, il exécute localement `qwen3:4b-instruct` sur le Mac.

Lors d'un appel simple :

```bash
ollama run qwen3:4b
```

le modèle répond à partir de ses poids et du contexte fourni dans le prompt. Il ne consulte pas automatiquement Internet et il ne lit pas automatiquement notre corpus HLM.

Le RAG sert précisément à apporter les bonnes informations au modèle :

```text
Question utilisateur
        ↓
Retrieval dans le corpus local
BM25 + dense + RRF + reranker
        ↓
Passages Service-Public / LEGI pertinents
        ↓
Qwen3:4b via Ollama
        ↓
Réponse fondée sur les passages + citations
```

Ainsi, pour une question sur le DALO, Qwen ne doit pas être utilisé uniquement sur sa mémoire interne : le système récupère d'abord les passages officiels pertinents puis les injecte dans son contexte.

## Arborescence utile

```text
Projet_Generative_AI/
├── README.md
├── pyproject.toml
├── uv.lock
├── Docs/
│   ├── CONTEXT.md
│   ├── AVANCEMENT.md
│   └── spec/
├── notebooks/
│   └── Assistant_HLM_RAG.ipynb
├── data/
│   ├── raw/
│   │   ├── service_public/
│   │   ├── legi/
│   │   ├── rpls/
│   │   ├── dvf/
│   │   └── plafonds/
│   ├── processed/
│   │   ├── service_public_hlm_documents.jsonl
│   │   ├── legi_hlm_documents.jsonl
│   │   ├── rag_documents.jsonl
│   │   └── chunks/
│   │       └── fixed_300_60.jsonl
│   └── indexes/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── tools/                # eligibility, social_housing (RPLS), market_prices (DVF), communes (COG)
│   ├── agent/                # schemas, ollama, rag, agent, baseline, retrieval, tool_agent, llm, tools_registry, guards
│   ├── config.py             # chemins fixes
│   └── settings.py           # réglages typés (configs/*.yaml + .env)
├── configs/                  # agent.yaml, tools.yaml
├── prompts/                  # system.md.j2, final_answer.md.j2, tool_descriptions.yaml
├── eval/
│   ├── golden.jsonl          # 52 questions, splits dev / test
│   ├── check_golden.py
│   ├── run_eval.py           # évaluation end-to-end
│   ├── metrics.py
│   ├── eval_bm25_chunking.py
│   ├── eval_dense.py
│   ├── eval_retrieval.py
│   ├── retrieval/            # summary.csv, details.csv (dev)
│   └── runs/                 # un dossier par run end-to-end
├── tests/
│   ├── test_retrieval.py
│   ├── test_eligibility.py
│   ├── test_eval_pipeline.py
│   ├── test_tools.py
│   ├── test_tool_security.py
│   └── test_agent.py
├── .env.example
└── app/
    └── streamlit_app.py
```

## Code Python principal

### Ingestion

```text
src/ingestion/service_public.py
src/ingestion/legi.py
```

Ces modules lisent et normalisent les sources documentaires.

### Prétraitement

```text
src/preprocessing/build_service_public_hlm_corpus.py
src/preprocessing/build_legi_hlm_corpus.py
src/preprocessing/build_rag_corpus.py
src/preprocessing/build_chunks.py
```

Ils construisent les corpus finaux et les différentes stratégies de chunking.

### Retrieval

```text
src/retrieval/config.py     # toute la configuration (chemins, modèles, k RRF, profondeurs)
src/retrieval/common.py     # lecture JSONL / chunks, hash, tri des scores
src/retrieval/bm25.py       # BM25Retriever
src/retrieval/dense.py      # DenseRetriever E5 + cache des embeddings
src/retrieval/hybrid.py     # fusion RRF
src/retrieval/reranker.py   # CrossEncoderReranker
src/retrieval/schemas.py    # RetrievedChunk, SearchResponse (Pydantic)
src/retrieval/search.py     # search_docs, warmup
```

Sur le MacBook Pro M1, PyTorch utilise **MPS** pour le dense et le reranker. Voir la section [Retrieval](#retrieval) pour les résultats et le contrat de `search_docs`.

## Données

### Données brutes

```text
data/raw/
```

Le ZIP Service-Public versionné est :

```text
data/raw/service_public/vosdroits-latest.zip
```

Les fichiers bruts volumineux LEGI, RPLS et DVF ne doivent pas nécessairement être versionnés dans Git.

### Données traitées

Les fichiers principaux sont :

```text
data/processed/service_public_hlm_documents.jsonl
data/processed/legi_hlm_documents.jsonl
data/processed/rag_documents.jsonl
data/processed/chunks/fixed_300_60.jsonl
```

`rag_documents.jsonl` contient les **455 documents** avant chunking.

`fixed_300_60.jsonl` contient les **1 357 chunks** actuellement utilisés pour l'évaluation du retrieval.

## Évaluation

Le golden set actif est :

```text
eval/golden.jsonl
```

Il contient **52 questions**, réparties en deux splits :

| Split | Questions | Rôle |
|---|---:|---|
| `dev` | 20 (q001-q020) | ont servi aux choix du retrieval (chunking, reranker) |
| `test` | 32 (q021-q052) | **held-out** : jamais utilisées pour régler le système, résultats finaux |

| Type | dev | test | Ce qui est vérifié |
|---|---:|---:|---|
| `rag` (Service-Public) | 18 | 10 | retrieval + citations ; les 10 du test sont en langage courant, dont 6 « jumeaux » de questions dev (`twin_of`) |
| `rag` (LEGI, CCH) | 0 | 7 | articles du Code de la construction (74 % du corpus) |
| `eligibility` | 0 | 5 | appel de `check_eligibility` et résultat exact (zone, catégorie, PLAI/PLUS/PLS) |
| `clarification` | 0 | 2 | l'assistant demande la commune et le foyer au lieu de conclure |
| `rpls`, `dvf` | 0 | 2 | `status: pending_data` tant que les outils ne sont pas prêts |
| `out_of_scope` | 1 | 2 | abstention sans chiffre inventé |
| `prompt_injection` | 1 | 2 | injection directe, injection dans la question, citation inventée |
| `causal_trap` | 0 | 2 | aucune conclusion causale HLM / délinquance / pauvreté |

Champs : `id`, `split`, `type`, `question`, `answer`, `gold_docs` (documents qui contiennent la réponse), `acceptable_docs` (autres documents qui la contiennent aussi : comptés pour Hit@k et MRR, pas pour Recall), `expected_tool`, `should_abstain`, `expected` pour l'éligibilité et `twin_of` pour les jumeaux de questions dev.

Contrôle d'intégrité (à lancer avant chaque commit du golden set) :

```bash
uv run python eval/check_golden.py --bm25
```

Avec BM25 seul, le split `dev` donne Hit@5 = 1,00 (18 questions) et le split `test` Hit@5 = 0,47 (17 questions) : le split `test` mesure enfin ce qu'apportent le dense et le reranker. Les paires `twin_of` comparent un même sujet formulé comme le titre de la fiche (dev) ou en langage courant (test).

Évaluations du retrieval :

```bash
uv run python eval/eval_bm25_chunking.py      # split dev
uv run python eval/eval_dense.py              # split dev
uv run python eval/eval_retrieval.py          # split dev -> eval/retrieval/
uv run python eval/eval_retrieval.py test     # split test -> eval/retrieval_test/
```

### Évaluation end-to-end

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

Options : `--backend baseline|rag|agent|agent_tools` (défaut `agent`), `--provider openrouter|ollama` (pour `agent_tools`), `--retriever hybrid|hybrid_norerank|bm25`, `--model qwen3:4b-instruct`, `--judge` (juge LLM ; `--judge-provider ollama|openrouter`, `--judge-model`), `--split dev|test`, `--run-name <nom>` (copie dans `eval/runs/<nom>/`), `--resume`, `--compare` (tableau de tous les runs dans `eval/runs/comparison.md`).

`eval/results.jsonl` contient, par question : réponse, sources citées, documents récupérés, tokens in/out, latence (totale et par étape), appels d'outils, stop reason, coût et métriques (`hit_at_5`, `recall_at_5`, `rr`, `citation_precision`, `cites_relevant_doc`, `invented_citation`, `abstention_ok`, `routing_ok`, `eligibility_exact`, `judge_score`). `eval/summary.md` agrège par split et par type.

## Interface

```bash
uv run streamlit run app/streamlit_app.py
```

Onglet **Assistant** : réponse, badge de statut (réponse sourcée, abstention, précision demandée, erreur), calcul d'éligibilité détaillé, sources cliquables, latence / tokens / coût, et trace de l'agent (outils appelés, arguments, passages avec rangs BM25 / dense / reranker). Onglet **Évaluation** : comparaison des runs de `eval/runs/` et détail question par question. Sans Ollama : backend `baseline` + recherche `BM25 seul`.

## Étapes restantes principales

Terminé : retrieval (`search_docs`), RAG modulaire (`src/agent/rag.py`), agent local par formulaire JSON (`src/agent/agent.py`), agent à tool calling natif (`src/agent/tool_agent.py`, OpenRouter ou Ollama), outils déterministes `check_eligibility`, `social_housing_stats` (RPLS) et `market_prices` (DVF), golden set de 52 questions, `eval/run_eval.py`, interface Streamlit. Restent :
1. compléter les réponses de q036 / q037 maintenant que RPLS et DVF sont intégrés ;
2. lancer les runs finaux (baseline, rag, agent local, agent à outils, 2 modèles) et rédiger l'analyse d'erreurs ;
3. préparer la démonstration, enregistrer la vidéo de secours et poser le tag `v1.0`.

Le suivi détaillé est maintenu dans [Docs/AVANCEMENT.md](Docs/AVANCEMENT.md).  
Les règles, décisions et contraintes durables sont dans [Docs/CONTEXT.md](Docs/CONTEXT.md).


## Travail collaboratif avec Git

Le projet est développé à plusieurs. Pour éviter qu'un membre de l'équipe n'écrase involontairement le travail déjà présent sur `main`, **aucun développement ne doit être réalisé directement sur `main`**.

Les branches personnelles de travail sont :

```text
main
├── charles
├── theodore
├── alexis
├── robin
└── gabby
```

Les noms de branches restent sans accent pour éviter les problèmes de compatibilité dans les commandes Git.

### Règle principale

`main` représente la version commune et stable du projet.

Chaque membre travaille sur sa propre branche, pousse ses modifications sur cette branche, puis ouvre une **Pull Request vers `main`**. Il ne faut pas utiliser `git push origin main` pour envoyer directement du développement sur la branche commune, et il ne faut pas utiliser `git push --force` sur `main`.

### Commencer une session de travail

Avant de commencer une nouvelle tâche, récupérer d'abord la dernière version de `main` :

```bash
git switch main
git pull --ff-only origin main
```

Puis revenir sur sa branche personnelle et y intégrer la dernière version de `main`.

Pour Charles :

```bash
git switch charles
git merge main
```

Pour les autres membres, remplacer `charles` par `theodore`, `alexis`, `robin` ou `gabby`.

Avant de modifier des fichiers, vérifier la branche active :

```bash
git branch --show-current
```

La commande doit afficher la branche personnelle du membre, et non `main`.

### Enregistrer son travail

Pendant le développement :

```bash
git status
git add .
git commit -m "Description claire de la modification"
git push origin charles
```

Charles utilise `charles`. Les autres membres poussent respectivement vers leur propre branche.

Exemples :

```bash
git push origin theodore
git push origin alexis
git push origin robin
git push origin gabby
```

### Intégrer le travail dans main

Une fois une tâche terminée :

1. pousser la branche personnelle sur GitHub ;
2. ouvrir une **Pull Request** ;
3. choisir `main` comme branche de destination (*base*) ;
4. choisir la branche personnelle comme branche source (*compare*) ;
5. relire les modifications et résoudre les éventuels conflits ;
6. fusionner la Pull Request seulement lorsque les changements sont validés.

Exemple :

```text
base:    main
compare: charles
```

Après la fusion de la Pull Request, mettre à jour sa copie locale :

```bash
git switch main
git pull --ff-only origin main
git switch charles
git merge main
```

### Si plusieurs personnes modifient les mêmes fichiers

Git ne remplace pas silencieusement deux versions incompatibles : il peut signaler un **conflit de fusion**. Dans ce cas, ne pas choisir une version au hasard et ne pas forcer le push. Examiner les deux modifications, conserver le contenu nécessaire, tester le projet, puis terminer la fusion.

### Commandes à éviter sur main

Sauf opération volontaire et maîtrisée d'administration du dépôt, ne pas utiliser :

```bash
git push --force origin main
git push -f origin main
git reset --hard origin/main
```

Un tag de sécurité peut être créé avant une étape importante afin de conserver un point de restauration de l'état du projet.

## Sécurité et versionnement

- ne jamais committer de clé API ;
- utiliser un fichier `.env` local et git-ignoré lorsque nécessaire ;
- fournir un `.env.example` avant la livraison finale ;
- ne pas committer les caches de modèles ou index lourds ;
- considérer les documents récupérés par le RAG comme des **données**, jamais comme des instructions ;
- valider les arguments des futurs appels d'outils avant exécution ;
- mesurer les cas de prompt injection et les stratégies d'abstention.

## Échéances

- **Code freeze** : 20 octobre 2026 à 23 h 59 — tag `v1.0`
- **Soutenance** : 21 octobre 2026 — 10 minutes de pitch + 5 minutes de questions
