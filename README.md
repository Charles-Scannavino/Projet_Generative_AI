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

Deux premières méthodes ont été évaluées sur les **18 questions RAG** du golden set :

| Méthode | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| BM25 | 1.0000 | 1.0000 | 0.8306 |
| Dense — `intfloat/multilingual-e5-small` | 0.9444 | 1.0000 | 0.7394 |

BM25 est donc meilleur seul sur le golden set actuel. La prochaine étape est la fusion **BM25 + dense par Reciprocal Rank Fusion (RRF)**, puis l'ajout d'un **reranker cross-encoder**.

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
│   ├── eval_bm25_chunking.py
│   └── eval_dense.py
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
src/retrieval/dense.py
```

Le retriever dense utilise actuellement :

```text
intfloat/multilingual-e5-small
```

avec des embeddings de **384 dimensions**. Sur le MacBook Pro M1, PyTorch utilise **MPS**.

Les prochains modules prévus sont notamment :

```text
src/retrieval/bm25.py
src/retrieval/hybrid.py
src/retrieval/reranker.py
```

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
```

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

1. implémenter la fusion hybride BM25 + dense avec **RRF** ;
2. ajouter un **reranker cross-encoder** ;
3. créer `search_docs` ;
4. connecter l'API locale Ollama depuis Python et intégrer `qwen3:4b` au RAG ;
5. ajouter citations, abstention et sorties validées avec Pydantic ;
6. intégrer les plafonds HLM et `check_eligibility` ;
7. intégrer RPLS et `social_housing_stats` ;
8. intégrer DVF et `market_prices` ;
9. construire l'agent et mesurer le routage des outils ;
10. étendre le golden set vers environ 40 questions ;
11. créer `eval/run_eval.py` et l'évaluation finale ;
12. préparer la démonstration et le tag `v1.0`.

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
