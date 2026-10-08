# Avancement — Projet Assistant HLM RAG & Agent

> Ce fichier est le **journal opérationnel** du projet. Il doit être mis à jour après chaque tâche significative. `CONTEXT.md` contient les règles et décisions durables ; `AVANCEMENT.md` contient l'état réel, les résultats, les problèmes rencontrés et les prochaines actions.

## 1. État global

**Statut actuel :** 🔍 Sélection du corpus Service-Public terminée à titre provisoire ; extraction du texte complet à réaliser.

**Étape en cours :**
- le corpus brut Service-Public a été parsé dans Colab ;
- 3 004 fiches `F*.xml` ont été indexées dans un DataFrame de métadonnées ;
- 350 fiches rattachées au sujet `Logement` ont été retenues comme **corpus Service-Public de travail** ;
- 505 fiches apparaissent dans le filtre logement élargi, mais ce filtre contient des faux positifs et n'est pas retenu tel quel ;
- 24 fiches correspondent au filtre HLM strict ;
- la prochaine étape est d'extraire et contrôler le **texte complet** des 350 fiches de travail, avant d'intégrer LEGI.

**Échéances :**
- gel du code : **20 octobre 2026 à 23 h 59** ;
- soutenance : **21 octobre 2026**.

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
- [ ] Préparer l'environnement local final avec `uv`.
- [ ] Vérifier l'exécution locale du projet sur laptop.
- [ ] Mettre en place Ollama / modèle local.

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
- [ ] Inspecter plus finement les 350 titres / contenus pour exclure les fiches manifestement hors périmètre si nécessaire.
- [ ] Extraire le texte complet des 350 fiches retenues.
- [ ] Nettoyer et normaliser les textes.
- [ ] Contrôler les documents vides, anormalement courts ou anormalement longs.
- [ ] Sauvegarder le corpus textuel Service-Public final dans `data/processed/`.

### LEGI

- [ ] Identifier précisément le sous-ensemble du Code de la construction et de l'habitation utile au projet.
- [ ] Télécharger / récupérer les articles.
- [ ] Définir l'unité documentaire.
- [ ] Parser les métadonnées et le texte.
- [ ] Ajouter les documents LEGI au corpus final.
- [ ] Vérifier que l'ensemble Service-Public + LEGI est cohérent avec la cible d'environ 500 documents.

### Chunking documentaire

- [ ] Lire manuellement un échantillon d'au moins 20 documents.
- [ ] Définir plusieurs stratégies de chunking.
- [ ] Tester au moins :
  - découpage par sections ;
  - taille fixe raisonnable avec overlap ;
  - variante adaptée à la structure Service-Public / LEGI.
- [ ] Conserver les métadonnées dans chaque chunk.
- [ ] Comparer les stratégies sur le golden set.
- [ ] Figer le chunking final.

> ⚠️ Le « chunk 22 » mentionné dans la conversation désignait une **cellule du notebook**, pas un chunk documentaire RAG. Le vrai chunking reste à faire.

### Golden set

- [x] Brouillon initial de 5 questions créé.
- [x] Golden set actif enrichi à **20 questions** dans `eval/golden.jsonl`.
- [x] Répartition actuelle : **18 RAG**, **1 hors périmètre**, **1 prompt injection**.
- [x] Les questions RAG sont ancrées sur des identifiants Service-Public réels.
- [x] Les cas d'éligibilité, RPLS et DVF non encore vérifiables sont conservés dans `eval/golden_backlog_pending.jsonl`.
- [ ] Étendre progressivement vers environ 40 questions après intégration de LEGI, des plafonds, de RPLS et de DVF.
- [ ] Couvrir :
  - factuel RAG ;
  - éligibilité ;
  - RPLS ;
  - DVF ;
  - absence de réponse ;
  - prompt injection ;
  - pièges causaux ;
  - abstention.
- [ ] Vérifier les `gold_docs` avec les identifiants réels du corpus.
- [ ] Ne pas remplir artificiellement les réponses ou sources avant validation des données.

### Retrieval

- [ ] Implémenter BM25.
- [ ] Mesurer BM25 sur le golden set.
- [ ] Implémenter le dense retrieval.
- [ ] Choisir et documenter le modèle d'embedding.
- [ ] Comparer dense et BM25.
- [ ] Implémenter la fusion hybride.
- [ ] Tester une méthode de fusion, par exemple RRF.
- [ ] Ajouter un reranker.
- [ ] Comparer :
  - BM25 ;
  - dense ;
  - hybride ;
  - hybride + reranker.
- [ ] Mesurer `Recall@5`, `Recall@10` et `MRR`.
- [ ] Comparer avec la baseline DINUM si possible.

### RAG / génération

- [ ] Construire `search_docs`.
- [ ] Ajouter le modèle génératif local.
- [ ] Produire des réponses fondées uniquement sur les sources récupérées.
- [ ] Ajouter les citations.
- [ ] Mettre en place une stratégie d'abstention.
- [ ] Valider les sorties avec Pydantic.

### Outils déterministes

- [ ] Télécharger les plafonds PLAI / PLUS / PLS.
- [ ] Implémenter `check_eligibility`.
- [ ] Écrire des tests unitaires sur plusieurs configurations de foyer.
- [ ] Télécharger / préparer RPLS.
- [ ] Implémenter `social_housing_stats`.
- [ ] Télécharger / préparer DVF.
- [ ] Implémenter `market_prices`.
- [ ] Toujours renvoyer l'année et la source des données.

### Agent

- [ ] Définir les descriptions des outils.
- [ ] Implémenter le routage de l'agent.
- [ ] Vérifier que :
  - question documentaire → `search_docs` ;
  - éligibilité → `check_eligibility` ;
  - statistiques HLM → `social_housing_stats` ;
  - marché immobilier → `market_prices`.
- [ ] Tracer les appels d'outils.
- [ ] Mesurer l'exactitude du routage.

### Évaluation complète

- [ ] Créer `eval/run_eval.py`.
- [ ] Respecter la commande demandée :
  ```bash
  uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
  ```
- [ ] Produire `eval/results.jsonl`.
- [ ] Enregistrer par question :
  - answer ;
  - cited sources ;
  - retrieved document IDs ;
  - tokens in/out ;
  - latency ;
  - tool calls ;
  - stop reason ;
  - coût.
- [ ] Évaluer la justesse finale.
- [ ] Analyser au moins 5 erreurs en détail.
- [ ] Comparer qualité / coût / latence.

### Interface et livraison

- [ ] Créer une interface simple Streamlit si le temps le permet.
- [ ] Afficher réponse, sources et trace de l'agent.
- [x] README initial du dépôt rédigé et mis à jour avec l'architecture, l'état du projet et le workflow.
- [ ] Documenter les versions exactes des modèles lorsque les choix seront figés.
- [ ] Ajouter `.env.example`.
- [x] `.gitignore` adapté au projet (Python/Colab, données extraites, index, résultats d'évaluation, secrets).
- [ ] Vérifier qu'aucune clé API n'est présente.
- [ ] Préparer la démonstration de secours enregistrée.
- [ ] Créer le tag `v1.0` au gel du code.

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

## 5. Prochaine action immédiate

La prochaine étape est désormais **l'extraction du texte complet des 350 fiches Service-Public de travail**.

Ordre conseillé :

1. écrire une fonction robuste d'extraction du contenu utile des XML ;
2. tester cette fonction sur 5 à 10 fiches connues, dont `F10007`, `F18005`, `F12006` ;
3. vérifier que le texte extrait conserve les titres de sections et ne duplique pas excessivement le contenu ;
4. appliquer la fonction aux 350 fiches ;
5. calculer la longueur de chaque document ;
6. repérer les textes vides, très courts ou anormalement longs ;
7. sauvegarder le corpus textuel propre dans `data/processed/`.

Ensuite :
- intégrer LEGI ;
- lire manuellement au moins 20 documents ;
- définir et tester le vrai **chunking documentaire RAG** ;
- commencer BM25 puis l'évaluation retrieval.

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

---

*Dernière mise à jour : 8 octobre 2026 — migration et structuration GitHub.*
