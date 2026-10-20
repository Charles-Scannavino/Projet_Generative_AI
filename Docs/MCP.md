# Serveur MCP — Assistant HLM

## 1. Présentation

Le serveur MCP permet à un client externe d'utiliser les quatre outils de l'assistant HLM : recherche documentaire, éligibilité, statistiques RPLS et statistiques DVF.

- Serveur : assistant-hlm
- Implémentation : src/mcp_server.py
- Transport : stdio
- SDK : MCP Python 2.x
- Outils exposés : search_docs, check_eligibility, social_housing_stats et market_prices
- Responsable : Charles

## 2. Architecture

Le serveur réutilise le moteur de recherche du projet : BM25, embeddings E5, fusion RRF et reranker MiniLM.

Le corpus contient 455 documents Service-Public et LEGI, découpés en 1 357 passages.

Le serveur retourne des résultats structurés avec sources et avertissements. Les calculs réglementaires et statistiques sont effectués par les modules Python existants, sans génération par un LLM.

## 3. Prérequis

- Git et accès au dépôt privé.
- Python 3.12 ou version compatible.
- uv pour installer les dépendances.
- Connexion Internet au premier chargement des modèles Hugging Face.
- Node.js uniquement pour la démonstration MCP Inspector.

Ollama et OpenRouter ne sont pas nécessaires au fonctionnement du serveur MCP.

## 4. Installation du projet

Cloner le dépôt sur la branche intégrant le MCP :

    git clone https://github.com/Charles-Scannavino/Projet_Generative_AI.git
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

Les neuf tests vérifient la découverte des quatre outils, leurs sorties réelles et les principaux cas d'erreur.

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

### Exemples des quatre appels dans MCP Inspector

Sélectionner l'outil dans l'onglet Tools et renseigner
les paramètres ci-dessous. Le mode « Edit as JSON » permet
également de saisir directement les arguments.

**Recherche documentaire DALO :**

~~~json
{"query": "Qu'est-ce que le DALO ?", "k": 5}
~~~

Résultat de référence : fiche Service-Public F18005,
méthode `hybrid_rrf+rerank`.

**Éligibilité HLM à Lyon :**

~~~json
{
  "revenu_fiscal_reference": 28000,
  "nb_personnes": 3,
  "nb_personnes_a_charge": 1,
  "commune": "Lyon",
  "annee": 2026
}
~~~

Résultat : PLAI non éligible, PLUS et PLS éligibles.
Les calculs reposent sur le barème 2026 et le RFR 2024.

**Parc locatif social à Créteil :**

~~~json
{"commune": "Créteil"}
~~~

Résultat : 15 133 logements recensés au 1er janvier 2025.
Cela ne représente pas les logements actuellement disponibles.

**Prix de vente à Villeurbanne :**

~~~json
{
  "commune": "Villeurbanne",
  "type_local": "Appartement",
  "annee": 2025
}
~~~

Résultat : prix médian 3 567 EUR/m2 sur 1 890 ventes.
DVF décrit les ventes immobilières et non les loyers.

## 8. Contrats des quatre outils

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

### check_eligibility

Paramètres : revenu_fiscal_reference (euros), nb_personnes,
nb_personnes_a_charge, jeune_menage, personne_handicapee,
commune ou zone, departement facultatif, annee (2026).

Fournir exactement une commune ou une zone, jamais les deux.
Le revenu est le revenu fiscal de référence N-2 du ménage.
Résultats : plafonds PLAI/PLUS/PLS, verdicts, écarts,
catégorie du foyer, provenance et avertissements.

### social_housing_stats

Paramètres : commune et departement facultatif.
Résultats : statistiques du parc social, financement initial,
caractéristiques, millésime RPLS et provenance SDES.

Le RPLS est un instantané annuel, pas une liste de logements
actuellement disponibles. Il ne permet pas ici de calculer
le taux de vacance ou le taux SRU.

### market_prices

Paramètres : commune, departement facultatif,
type_local (Appartement, Maison ou Tous),
annee facultative.

Résultats : nombre de ventes, médiane du prix au m2,
quartiles, années disponibles, provenance DGFiP.
DVF décrit des prix de vente et non des loyers.
Les statistiques portant sur moins de cinq ventes sont masquées.

## 9. Gestion des erreurs

Le serveur vérifie les paramètres des quatre outils via MCP et les modèles Pydantic stricts.

Une requête vide ou une valeur k hors limites est rejetée.

Exemples de requêtes invalides :

    {"query": "", "k": 5}
    {"query": "DALO", "k": 0}

Les erreurs de validation sont converties en erreurs MCP
contrôlées avec ToolError.

Les tests automatisés vérifient aussi les compositions de foyer incohérentes, années non couvertes, communes inconnues et types de biens invalides.

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

### 20 octobre 2026 — Quatre outils MCP validés

- Quatre outils découverts dans MCP Inspector v2.11.0.
- Protocole MCP 2025-11-25, transport stdio.
- search_docs : DALO, document F18005 en première position,
  hybrid_rrf+rerank, environ 9,39 secondes.
- check_eligibility : Lyon, trois personnes, RFR 28 000 EUR ;
  PLAI non, PLUS et PLS oui ; environ 41 ms.
- social_housing_stats : Créteil, 15 133 logements sociaux
  au 1er janvier 2025 ; environ 41 ms.
- market_prices : Villeurbanne, appartements, 2025,
  médiane de 3 567 EUR/m2 sur 1 890 ventes ; environ 1,86 s.
- Chaque outil testé renvoie des données structurées ;
  les trois outils métier renvoient sources et avertissements.
- Neuf tests MCP réussis.
- Suite générale : 195 tests réussis, un désélectionné.
- Golden set : 52 questions valides.

Les latences correspondent aux essais affichés dans Inspector
et ne représentent pas des moyennes sur plusieurs exécutions.

## 11. Limites du serveur MCP

- Les quatre outils sont exposés via MCP.
- Le serveur ne génère pas lui-même de réponse en langage naturel.
- Les appels depuis MCP Inspector sont validés ; aucune intégration
  autonome avec un agent IA externe n'a encore été démontrée.
- RPLS ne fournit pas les disponibilités de logements en temps réel.
- DVF porte sur les ventes, pas les loyers.
- Seul le barème d'éligibilité 2026 est couvert.
- Le premier chargement du moteur RAG peut être lent.
- Le transport est stdio, pas HTTP.
- Un client MCP compatible est nécessaire.

### Perspective — Connexion à Claude Desktop

Un client MCP peut utiliser les outils de ce serveur sans
passer par l'interface Streamlit.

Une intégration future à Claude Desktop consisterait à :

1. Installer et ouvrir Claude Desktop sur macOS.
2. Configurer un serveur MCP local `assistant-hlm` dans
   `claude_desktop_config.json`.
3. Utiliser le chemin absolu de `uv` et le chemin du dépôt.
4. Redémarrer Claude Desktop et vérifier les outils disponibles.
5. Poser une question HLM et autoriser l'appel à l'outil adapté.
6. Contrôler le résultat, les sources et les avertissements.

Sur le Mac de référence, l'exécutable uv se trouve dans
`/opt/homebrew/bin/uv`. Le répertoire du dépôt dépend
de l'utilisateur et doit être configuré en chemin absolu.

Exemple de configuration à adapter :

~~~json
{
  "mcpServers": {
    "assistant-hlm": {
      "command": "/opt/homebrew/bin/uv",
      "args": [
        "--directory",
        "/CHEMIN/ABSOLU/Projet_Generative_AI",
        "run",
        "python",
        "-m",
        "src.mcp_server"
      ]
    }
  }
}
~~~

Cette configuration est illustrative et non encore validée
dans Claude Desktop. Si l'environnement uv est externe au
projet, il faut aussi transmettre UV_PROJECT_ENVIRONMENT
via la configuration du client.

Un assistant externe pourra ainsi sélectionner un outil,
l'appeler et générer une réponse conversationnelle avec
les données retournées par notre serveur.

Attention : les données partagées avec un assistant cloud
peuvent être transmises à son fournisseur. Éviter les
informations personnelles ou confidentielles lors des tests.

Le branchement d'un client IA externe n'a pas encore été testé.
La validation actuelle porte sur le protocole et les quatre
outils via MCP Inspector.

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

La version initiale du serveur a été fusionnée dans main via la PR #4 (commit 8c0e567).

L'extension aux quatre outils est développée sur charles-bonus-mcp-complet. Sa publication dans main doit être vérifiée après la fusion.
