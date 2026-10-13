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

Le modèle de départ retenu est :

```bash
ollama pull qwen3:4b
```

Vérifier les modèles installés :

```bash
ollama list
```

La configuration validée affiche actuellement :

```text
qwen3:4b    2.5 GB
```

### 6. Tester le modèle

Lancer une session interactive :

```bash
ollama run qwen3:4b
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

## Ollama : rôle dans le projet

Ollama **n'est pas un moteur de recherche Web**. Dans notre configuration, il exécute localement `qwen3:4b` sur le Mac.

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
│   ├── tools/
│   └── agent/
├── eval/
│   ├── golden.jsonl
│   ├── golden_backlog_pending.jsonl
│   ├── metrics.py
│   ├── eval_bm25_chunking.py
│   ├── eval_dense.py
│   ├── eval_retrieval.py
│   └── retrieval/            # summary.csv, details.csv
├── tests/
│   └── test_retrieval.py
└── app/
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

Il contient actuellement :

- **18 questions RAG** ;
- **1 question hors périmètre** ;
- **1 prompt injection**.

Les futurs cas d'éligibilité, RPLS et DVF sont conservés dans :

```text
eval/golden_backlog_pending.jsonl
```

Évaluations déjà disponibles :

```bash
uv run python eval/eval_bm25_chunking.py
uv run python eval/eval_dense.py
uv run python eval/eval_retrieval.py
```

`eval/eval_retrieval.py` est l'ablation unifiée du retrieval (BM25, dense, hybride, hybride + rerankers) ; ses résultats sont écrits dans `eval/retrieval/`. Les tests du contrat `search_docs` se lancent avec `uv run pytest -q`.

La commande finale prévue par l'énoncé sera :

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

**Attention : `eval/run_eval.py` n'est pas encore implémenté à ce stade.**

À terme, l'évaluation finale doit enregistrer par question au minimum :

- réponse ;
- sources citées ;
- documents récupérés ;
- tokens en entrée/sortie ;
- latence ;
- appels d'outils ;
- raison d'arrêt / abstention ;
- coût.

## Étapes restantes principales

La fusion hybride RRF, le reranker et `search_docs` sont terminés (voir [Retrieval](#retrieval)). Restent :

1. connecter l'API locale Ollama depuis Python et intégrer `qwen3:4b` au RAG ;
2. ajouter citations, abstention et sorties validées avec Pydantic ;
3. intégrer les plafonds HLM et `check_eligibility` ;
4. intégrer RPLS et `social_housing_stats` ;
5. intégrer DVF et `market_prices` ;
6. construire l'agent et mesurer le routage des outils ;
7. étendre le golden set vers environ 40 questions ;
8. créer `eval/run_eval.py` et l'évaluation finale ;
9. préparer la démonstration et le tag `v1.0`.

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
