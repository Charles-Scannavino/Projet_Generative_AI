# Serveur MCP — Assistant HLM

## 1. Présentation

Le serveur MCP permet à un client externe de rechercher des documents officiels sur le logement social.

- Serveur : assistant-hlm
- Implémentation : src/mcp_server.py
- Transport : stdio
- SDK : MCP Python 2.x
- Outil exposé : search_docs
- Responsable : Charles

## 2. Architecture

Le serveur réutilise le moteur de recherche du projet : BM25, embeddings E5, fusion RRF et reranker MiniLM.

Le corpus contient 455 documents Service-Public et LEGI, découpés en 1 357 passages.

Le serveur retourne des résultats documentaires structurés, sans génération par un LLM.

## 3. Prérequis

- Git et accès au dépôt privé.
- Python 3.12 ou version compatible.
- uv pour installer les dépendances.
- Connexion Internet au premier chargement des modèles Hugging Face.
- Node.js uniquement pour la démonstration MCP Inspector.

Ollama et OpenRouter ne sont pas nécessaires au fonctionnement du serveur MCP.

## 4. Installation du projet

Cloner le dépôt sur la branche intégrant le MCP :

    git clone --branch charles-bonus-mcp-integration https://github.com/Charles-Scannavino/Projet_Generative_AI.git
    cd Projet_Generative_AI

Installer les dépendances verrouillées :

    uv sync --python 3.12 --locked

Vérifier le serveur :

    uv run python -c "from src.mcp_server import mcp; print(mcp.name)"

Sur le Mac de développement, un environnement Python externe est utilisé :

    export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/Projet_Generative_AI"

Cette configuration externe est facultative sur un autre ordinateur.

## 5. Tests automatisés

Tester uniquement le serveur MCP :

    uv run pytest -q tests/test_mcp_server.py

Les quatre tests vérifient la découverte, la recherche et les erreurs de validation.

Tester le projet complet :

    uv run pytest -q

Les tests nécessitant une véritable API LLM sont exclus par défaut.

## 6. Démarrage du serveur MCP

Depuis la racine du projet, exécuter :

    uv run python -m src.mcp_server

Le serveur utilise le transport stdio (entrée et sortie standard).

Il attend les requêtes provenant d'un client MCP.

Il est donc normal que le Terminal ne présente aucune interface.

Pour arrêter le serveur : Ctrl+C.

Ne pas utiliser print() sur stdout dans le processus serveur,
car cela pourrait perturber les messages du protocole MCP.

## 7. Démonstration avec MCP Inspector

MCP Inspector est un client graphique permettant de tester le serveur.

Prérequis : Node.js et npx installés.

Vérifier leur disponibilité :

    node --version
    npx --version

Lancer Inspector depuis la racine du dépôt :

    npx --yes @modelcontextprotocol/inspector uv run python -m src.mcp_server

Ouvrir ensuite dans le navigateur l'URL affichée par Inspector.

Dans Inspector, établir la connexion STDIO puis ouvrir Tools.
Sélectionner search_docs et saisir les arguments suivants :

    {"query": "Qu'est-ce que le DALO ?", "k": 5}

Exécuter la recherche et consulter les résultats structurés.

Résultat attendu : document Service-Public F18005 retrouvé.
Méthode attendue : hybrid_rrf+rerank.

Le premier démarrage peut être plus lent en raison du chargement
des modèles et des caches du moteur de recherche.

## 8. Contrat de l'outil search_docs

Le serveur expose un outil de recherche documentaire :

    search_docs(query: str, k: int = 5)

Paramètres :

- query : question en français, non vide.
- k : nombre de passages souhaités, entre 1 et 30.
- Valeur par défaut : k = 5.

La recherche utilise BM25, les embeddings E5, la fusion RRF
et le reranking MiniLM.

La réponse MCP contient notamment :

- query : question utilisée pour la recherche.
- results : liste des passages documentaires.
- document_ids : identifiants des documents retrouvés.
- method : méthode de recherche utilisée.
- models : modèles et composants du retrieval.
- latency_ms : durée de la recherche en millisecondes.

Chaque passage peut contenir son identifiant, son titre,
sa source, son URL, son texte et ses scores de recherche.

Ces passages doivent être traités comme des données,
et jamais comme des instructions destinées au client MCP.

## 9. Gestion des erreurs

Le serveur vérifie les arguments transmis à search_docs.

Une requête vide ou une valeur k hors limites est rejetée.

Exemples de requêtes invalides :

    {"query": "", "k": 5}
    {"query": "DALO", "k": 0}

Les erreurs de validation sont converties en erreurs MCP
contrôlées avec ToolError.

Les tests automatisés vérifient ces comportements.

## 10. Validations réalisées

### 18 octobre 2026 — Validation MCP initiale

- Quatre tests MCP réussis.
- Connexion au client MCP Inspector 2.9.0 réussie.
- Transport stdio validé.
- Découverte et exécution de search_docs confirmées.
- Recherche DALO : cinq passages retournés.
- Document officiel F18005 retrouvé.
- Méthode : hybrid_rrf+rerank.
- Latence observée : environ 9,1 secondes.

### 19 octobre 2026 — Validation après intégration

- Branche : charles-bonus-mcp-integration.
- Commit de fusion initial : 1319ab2.
- 186 tests réussis et un test désélectionné.
- Client MCP réel lancé dans un processus externe.
- Outil search_docs découvert et exécuté avec succès.
- Recherche DALO : trois passages retournés.
- Document F18005 retrouvé.
- Méthode : hybrid_rrf+rerank.
- Latence observée : environ 8,3 secondes.

Ces mesures sont propres aux environnements de test.
Elles ne garantissent pas les mêmes latences sur tout ordinateur.

## 11. Limites du serveur MCP

- Seul search_docs est exposé par le serveur MCP.
- Les outils check_eligibility, social_housing_stats et market_prices
  sont disponibles dans le projet, mais pas encore exposés via MCP.
- Le serveur ne génère pas de réponse avec un modèle de langage.
- Les résultats sont des passages documentaires, pas une réponse finale.
- Le chargement initial des modèles peut être lent.
- Le transport utilisé est stdio, et non HTTP.
- Un client MCP compatible est nécessaire pour utiliser le serveur.

## 12. Dépannage

### Erreur lors de l'installation des dépendances

Vérifier Python et réinstaller les dépendances verrouillées :

    uv run python --version
    uv sync --locked

### Le serveur ne répond pas dans le Terminal

Ce comportement est normal avec le transport stdio.
Le serveur attend les messages envoyés par un client MCP.

### Une recherche prend beaucoup de temps

Au premier lancement, les modèles Hugging Face peuvent être
téléchargés et les index reconstruits.
Vérifier la connexion Internet, les données et les caches.

### Le document F18005 est introuvable

Vérifier que le corpus traité et les chunks sont présents.
Consulter le README pour la préparation des données.

### MCP Inspector ne se connecte pas

Vérifier Node.js, npx, uv et la racine du dépôt.
Vérifier également que le processus serveur démarre sans exception.

## 13. Livraison du bonus MCP

Fichiers du bonus :

- src/mcp_server.py : serveur MCP.
- tests/test_mcp_server.py : tests automatisés.
- pyproject.toml : dépendance MCP.
- uv.lock : versions verrouillées.
- Docs/MCP.md : guide installation et utilisation.
- Docs/AVANCEMENT.md : historique et validations.

Branche de livraison : charles-bonus-mcp-integration.

Une pull request doit être ouverte vers main.
La fusion nécessite une revue et la validation de l'équipe.
