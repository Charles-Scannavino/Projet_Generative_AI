# Contexte persistant — Projet Assistant HLM RAG & Agent

> Ce fichier constitue le **contexte de référence** du projet. Il doit être relu au début de chaque nouvelle session consacrée à ce projet et mis à jour si le périmètre, l'architecture, les données, l'environnement de travail ou les contraintes changent.

## 1. Informations générales

- **Projet** : Assistant HLM — RAG et agent sur données ouvertes françaises.
- **Dépôt GitHub de référence** : `Charles-Scannavino/Projet_Generative_AI`.
- **Branche principale** : `main`.
- **Cours** : Generative AI I — projet de groupe.
- **Équipe** : 3 à 4 étudiants.
- **Poids dans la note** : 50 % du cours.
- **Soutenance** : 21 octobre 2026 — 10 minutes de pitch + 5 minutes de questions.
- **Gel du code** : 20 octobre 2026 à 23 h 59, tag Git `v1.0`.
- **Objectif général** : construire un assistant capable d'aider un particulier sur le logement social :
  - comprendre ses droits et démarches ;
  - vérifier son éligibilité à un logement social ;
  - consulter des statistiques communales sur le parc social ;
  - consulter des informations sur le marché immobilier ;
  - répondre avec des sources et des citations ;
  - être évalué quantitativement sur un golden set.

## 2. Doctrine du projet

Le projet ne doit pas être présenté comme « fonctionnel » uniquement parce qu'une démonstration marche.

Chaque composant important doit être **mesuré** sur un jeu de questions de référence, avec au minimum :
- qualité du retrieval ;
- justesse de la réponse ;
- citations ;
- latence ;
- tokens entrants/sortants ;
- appels d'outils ;
- coût ;
- raison d'arrêt / abstention si applicable.

Le projet doit rester explicable : une architecture simple, mesurée et bien comprise vaut mieux qu'un système complexe difficile à justifier.

## 3. Périmètre fonctionnel

L'assistant doit combiner deux familles de capacités.

### 3.1 RAG documentaire

Sources principales :
- **Service-Public.fr — Particuliers**, principalement les fiches liées au logement et au logement social ;
- **LEGI — Code de la construction et de l'habitation**, notamment les articles utiles à l'attribution et au fonctionnement du logement social.

Le RAG doit permettre de répondre à des questions telles que :
- « Comment déposer une demande de logement social ? »
- « Qu'est-ce que le DALO ? »
- « Comment fonctionne le supplément de loyer de solidarité ? »
- « Quelles sont les règles applicables à un bail HLM ? »

### 3.2 Outils déterministes

Les données tabulaires ou les calculs réglementaires ne doivent pas être confiés directement au LLM.

Outils prévus :
- `check_eligibility` : vérifie l'éligibilité selon les revenus, le foyer, la zone et l'année ;
- `social_housing_stats` : interroge les données RPLS par commune ;
- `market_prices` : calcule des statistiques DVF, notamment le prix médian au m² ;
- `search_docs` : recherche dans le corpus documentaire.

Le LLM orchestre les outils mais ne remplace pas les calculs déterministes.

## 4. Sources de données prévues

### Service-Public — Particuliers
- Source textuelle principale du RAG.
- Fichier brut actuellement utilisé : `vosdroits-latest.zip`.
- Format : ensemble de fiches XML.
- Métadonnées utiles : identifiant, titre, sujet/thème, description, date, URL, contenu.

### LEGI
- Code de la construction et de l'habitation.
- À ajouter pour renforcer la couche juridique.
- L'unité documentaire devra être définie clairement, de préférence **un article = un document** lorsque c'est possible.

### RPLS
- Répertoire du parc locatif social.
- Utilisation : nombre de logements sociaux, logements loués, vacance, certaines caractéristiques du parc.
- Limite : photographie statistique annuelle, **pas de disponibilité en temps réel**.

### DVF
- Demandes de valeurs foncières.
- Utilisation : statistiques de prix de vente.
- Limite : DVF renseigne les **ventes**, pas les loyers.

### Plafonds de ressources
- Barèmes PLAI / PLUS / PLS.
- Utilisation : calcul d'éligibilité.
- L'année du barème doit toujours être conservée en métadonnée.

### DINUM / jeux vectorisés
- Les versions vectorisées existantes peuvent être utilisées comme **baseline**.
- Les choix de chunking et d'embeddings du projet doivent néanmoins être réalisés, justifiés et évalués par l'équipe.

### SSMSI / délinquance
- Optionnel.
- Ne pas établir de causalité entre logement social et délinquance.
- Si cette source n'apporte pas une valeur claire au projet, elle peut être supprimée du périmètre.

## 5. Architecture cible

Pipeline documentaire cible :

```text
Documents bruts
    ↓
Nettoyage
    ↓
Découpage en chunks + métadonnées
    ↓
BM25 + dense retrieval
    ↓
Fusion hybride
    ↓
Reranking
    ↓
Top-k chunks
    ↓
LLM
    ↓
Réponse structurée + citations
```

Architecture agentique cible :

```text
Utilisateur
    ↓
Agent
    ├── search_docs
    ├── check_eligibility
    ├── social_housing_stats
    └── market_prices
    ↓
Réponse structurée + sources + trace
```

La sortie finale doit être validée, idéalement avec **Pydantic**.

## 6. Évaluation

Le fichier attendu est :

```text
eval/golden.jsonl
```

Il doit contenir environ **40 questions**, avec plusieurs catégories :
- questions factuelles RAG ;
- questions d'éligibilité ;
- statistiques RPLS ;
- questions DVF ;
- questions sans réponse ;
- prompt injections ;
- questions pièges / causalité ;
- cas nécessitant une abstention.

Commande finale obligatoire :

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

Le script doit produire au minimum, pour chaque question :
- réponse ;
- sources citées ;
- identifiants des documents récupérés ;
- tokens in/out ;
- latence ;
- appels d'outils ;
- stop reason.

Métriques à suivre :
- `Recall@k` ;
- `MRR` ;
- justesse des réponses ;
- exactitude des citations ;
- exactitude du routage des outils ;
- latence ;
- coût.

Ablations attendues :
- BM25 ;
- dense ;
- hybride ;
- hybride + reranker ;
- comparaison de nos embeddings avec la baseline DINUM lorsque possible.

## 7. Environnement de travail actuel

### Prototypage
- **Langage** : Python.
- **Environnement actuel** : Google Colab, notebook `.ipynb`.
- **Persistance** : Google Drive pour les fichiers importants.
- Les fichiers temporaires volumineux ou les milliers de petits XML doivent être extraits dans `/content/` lorsque cela accélère fortement les traitements.

### Versionnement et source de vérité
- **GitHub est désormais la source de vérité** pour le code, le notebook, la documentation et les fichiers d'évaluation.
- Google Drive / le stockage local restent utilisés pour les données de travail volumineuses et l'exécution Colab.
- Avant toute modification importante, lire la version courante de `Docs/CONTEXT.md` et `Docs/AVANCEMENT.md` depuis le dépôt et vérifier l'état de la branche `main`.
- **Machine locale de référence** : MacBook Pro **M1** (Apple Silicon).
- **Homebrew (`brew`) est installé** sur la machine locale.
- L'environnement local du projet est maintenant géré avec **uv** et Python **3.12+**.
- `pyproject.toml` et `uv.lock` sont versionnés à la racine du dépôt.
- **Ollama est installé localement**. Modèle génératif : **`qwen3:4b-instruct`** (Qwen3-4B-Instruct-2507, 2,5 Go). Le tag `qwen3:4b` pointe vers la variante *Thinking*, qui raisonne toujours avant de répondre (cause probable des 67 à 138 s par question mesurées au départ).

### Finalisation
Le notebook sert au prototypage, à l'exploration et aux expériences.

La version finale devra être modularisée en fichiers `.py`, afin de permettre :
- une exécution reproductible ;
- l'évaluation via `eval/run_eval.py` ;
- le fonctionnement local sur un ordinateur portable ;
- l'utilisation d'Ollama ou de Transformers en local.

## 8. Arborescence cible

```text
Projet_Generative_AI/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example                    # à créer avant la livraison
├── .gitignore
├── Docs/
│   ├── CONTEXT.md
│   ├── AVANCEMENT.md
│   └── spec/
│       ├── project-instructions.md
│       ├── project-instructions.pdf
│       └── Assistant HLM RAG et agent sur données ouvertes.docx
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
│   └── indexes/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── tools/
│   └── agent/
├── eval/
│   ├── golden.jsonl
│   ├── check_golden.py
│   ├── run_eval.py
│   └── metrics.py
└── app/
    └── streamlit_app.py
```

Les dossiers de code encore vides sont conservés avec des fichiers `.gitkeep` jusqu'à leur implémentation.

## 9. État de référence actuel

À la date de cette mise à jour :
- le sujet HLM est défini ;
- le corpus Service-Public Particuliers a été choisi comme première source ;
- `vosdroits-latest.zip` a été téléchargé ;
- l'extraction directe de milliers de XML vers Google Drive s'est révélée lente ;
- la stratégie a été corrigée pour extraire les XML dans le stockage local Colab (`/content/`) ;
- les métadonnées Service-Public ont été explorées ;
- le filtre HLM strict retourne **24 fiches candidates** ;
- **350 fiches** dont le sujet contient `Logement` constituent le corpus Service-Public de travail ;
- un filtre logement élargi retourne **505 fiches**, avec des faux positifs ;
- le golden set actif contient désormais **20 questions** (18 RAG, 1 hors périmètre, 1 prompt injection) ;
- l'environnement local `uv` est initialisé avec `pyproject.toml` et `uv.lock` ;
- les dépendances de base sont installées : `lxml`, `numpy`, `pandas`, `pydantic`, `rank-bm25`, `tqdm` et `pytest` ;
- l'arborescence modulaire `src/ingestion`, `src/preprocessing`, `src/retrieval`, `src/tools` et `src/agent` existe ;
- LEGI, RPLS, DVF et les plafonds restent à intégrer ;
- le chunking RAG au sens documentaire n'a pas encore été finalisé ;
- BM25, dense retrieval, recherche hybride, reranker, agent et évaluation complète restent à construire.

Important : **ne pas confondre les numéros de cellules du notebook (« chunk 22 » dans la conversation) avec les chunks documentaires du RAG.**

## 10. Règles de travail à appliquer à chaque session

1. **Lire `CONTEXT.md` et `AVANCEMENT.md` au début d'une nouvelle session liée au projet.**
2. **Expliquer d'abord ce que l'on cherche à faire, pourquoi et comment ; proposer le code ensuite.**
3. Ne pas marquer une tâche comme terminée sans preuve concrète : sortie de cellule, fichier généré, métrique, test ou confirmation utilisateur.
4. Mettre à jour `AVANCEMENT.md` après chaque étape significative :
   - nouvelle source intégrée ;
   - nouveau parser ;
   - corpus figé ;
   - nouveau retriever ;
   - nouvelle métrique ;
   - nouvel outil ;
   - nouveau test ;
   - erreur importante et résolution.
5. Mettre à jour `CONTEXT.md` uniquement lorsque le contexte durable évolue : périmètre, architecture, environnement, règles, sources ou décisions structurantes.
6. Vérifier la cohérence entre `CONTEXT.md` et `AVANCEMENT.md`.
7. **Golden set tôt** : ne pas attendre la fin du développement pour créer l'évaluation.
8. **Calculs déterministes en Python** : éligibilité, agrégations statistiques et règles chiffrées ne doivent pas être calculées librement par le LLM.
9. **Citations obligatoires** pour les réponses documentaires.
10. **Local first** : la version finale doit fonctionner sur laptop avec un modèle local.
11. **Sécurité** : aucune clé API dans Git ; utiliser `.env` git-ignoré et fournir `.env.example`.
12. Si un dépôt Git est utilisé et accessible dans la session, synchroniser la branche de travail avant toute modification importante (`git fetch` puis `git pull`) et ne jamais écraser des changements locaux non vérifiés.
13. Préserver une approche pédagogique : expliquer les notions de RAG, retrieval, embeddings, reranking, métriques et agent afin que l'étudiant puisse les défendre à l'oral.

## 11. Limites et principes à rappeler dans les réponses de l'assistant final

- RPLS ne permet pas de dire qu'un logement est disponible aujourd'hui.
- DVF ne représente pas les loyers.
- Les plafonds de ressources dépendent de l'année et doivent être versionnés.
- Une corrélation statistique ne permet pas d'affirmer une causalité.
- L'assistant informe mais ne remplace pas l'ADIL, un bailleur, une mairie ou un professionnel compétent.
- Si le corpus ne permet pas de répondre de façon fiable, l'assistant doit savoir **s'abstenir**.

---

*Dernière mise à jour : 13 octobre 2026 — environnement local précisé : MacBook Pro M1, Homebrew, uv/Python.*
