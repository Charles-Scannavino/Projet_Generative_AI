# Avancement — Projet Assistant HLM RAG & Agent

> Ce fichier est le **journal opérationnel** du projet. Il doit être mis à jour après chaque tâche significative. `CONTEXT.md` contient les règles et décisions durables ; `AVANCEMENT.md` contient l'état réel, les résultats, les problèmes rencontrés et les prochaines actions.

## 1. État global

**Statut actuel :** ✅ Corpus Service-Public HLM final généré et contrôlé : **117 fiches conservées**, **233 exclues**, **0 document vide**, **0 doublon**, **15/15 `gold_docs` présents**. Prochaine étape : intégration de LEGI.

**Étape en cours :**
- le corpus brut Service-Public a été parsé dans Colab ;
- 3 004 fiches `F*.xml` ont été indexées dans un DataFrame de métadonnées ;
- 350 fiches rattachées au sujet `Logement` ont été retenues comme **corpus Service-Public de travail** ;
- 505 fiches apparaissent dans le filtre logement élargi, mais ce filtre contient des faux positifs et n'est pas retenu tel quel ;
- 24 fiches correspondent au filtre HLM strict ;
- un parser local `src/ingestion/service_public.py` a été créé et testé sur 5 fiches de référence ;
- les 350 fiches contenant `Logement` dans `dc:subject` ont été extraites dans `data/processed/service_public_documents.jsonl` ;
- la déduplication ciblée des conteneurs (`ASavoir`, `Attention`, `ANoter`, `Exemple`, `Definition`) a été validée sur les 5 fiches de référence sans perte du texte métier de `F21051` ;
- les 350 fiches ont été régénérées : 0 document vide, minimum 48 caractères / 8 mots, maximum 45 331 caractères / 7 498 mots, moyenne 9 872,5 caractères / 1 555,6 mots ;
- l'audit des 10 documents les plus courts et longs a été réalisé ;
- un échantillon reproductible de 20 documents (seed 42) a été lu et montre que le thème `Logement` est trop large pour l'assistant HLM : il contient aussi urbanisme, copropriété, piscines, taxe d'aménagement, forage, etc. ;
- audit hiérarchique réalisé sur les 350 fiches : 107 `Bail d’habitation`, 63 `Urbanisme`, 47 `Copropriété`, 40 `Aides financières pour le logement`, 31 `Vie pratique dans un logement`, 28 `Achat immobilier`, 26 `Protection et sécurité de l'habitat`, 8 `Expropriation - Préemption` ;
- dossiers particulièrement pertinents identifiés : `Accéder à un logement social` (8), `Aides personnelles au logement` (8), ainsi que plusieurs dossiers de bail d'habitation ;
- classification automatique du périmètre exécutée : **125 KEEP**, **33 REVIEW**, **192 EXCLUDE** ;
- revue manuelle terminée : les **33 `REVIEW`** ont été exclus du cœur HLM ; parmi les **13 `social_keyword`**, **5** ont été conservés et **8** exclus ;
- corpus Service-Public HLM final généré : **117 KEEP**, **233 EXCLUDE**, **0 REVIEW** ;
- contrôles finaux validés : **0 document vide**, **0 doublon**, **15/15 `gold_docs` présents**, aucun `gold_doc` manquant ;
- fichiers générés : `data/processed/service_public_hlm_documents.jsonl` et `data/processed/service_public_scope_decisions_final.csv` ;
- prochaine étape : intégrer LEGI afin de compléter le corpus documentaire avant les expérimentations de chunking.

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
- [x] Préparer l'environnement local final avec `uv` (`pyproject.toml`, `uv.lock`, Python 3.12.7 vérifié).
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
- [x] Inspecter un échantillon reproductible de 20 documents ainsi que les extrêmes de longueur ; constat : le thème `Logement` contient de nombreuses fiches hors cœur HLM.
- [x] Extraire le texte complet des 350 fiches retenues (première version du parser).
- [x] Nettoyer et normaliser les textes (normalisation de base + correction ciblée des redondances structurelles parent/enfant).
- [x] Contrôler les documents vides, anormalement courts ou anormalement longs (0 vide ; extrêmes inspectés).
- [x] Sauvegarder le corpus textuel Service-Public final dans `data/processed/` (`service_public_hlm_documents.jsonl`, **117 documents**).

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
- [x] Vérifier les `gold_docs` avec les identifiants réels du corpus (**15/15 présents**).
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

### Bonus évoqués par l'enseignant

En complément du périmètre principal, deux pistes peuvent apporter une valeur supplémentaire au projet : **exposer la solution via un serveur MCP**, afin de pouvoir brancher notre assistant et ses outils à différents LLM compatibles, et **préparer une mise en production simple**, de façon à ce qu'un utilisateur puisse lancer et utiliser la solution finale depuis son propre ordinateur sans dépendre de notre environnement de développement. Ces éléments sont considérés comme des bonus et ne doivent pas retarder les fonctionnalités principales du RAG, de l'agent et de l'évaluation.


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

Le corpus Service-Public est désormais **figé à 117 documents** après audit du périmètre et contrôles qualité. Les **15 `gold_docs`** du golden set actif sont présents.

La prochaine étape est **l'intégration de LEGI — Code de la construction et de l'habitation** :

1. identifier les articles réellement utiles au logement social ;
2. définir l'unité documentaire, idéalement **un article = un document** ;
3. récupérer et parser le texte avec ses métadonnées juridiques ;
4. contrôler les documents vides, doublons et métadonnées ;
5. fusionner LEGI avec les 117 fiches Service-Public ;
6. vérifier la cohérence du corpus documentaire global avec la cible d'environ 500 documents.

Après LEGI :
- définir et comparer plusieurs stratégies de **chunking documentaire** ;
- implémenter **BM25** ;
- mesurer `Recall@5`, `Recall@10` et `MRR` sur le golden set.

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

---

*Dernière mise à jour : 9 octobre 2026 — corpus Service-Public HLM final figé à 117 documents ; prochaine étape : intégration de LEGI.*
