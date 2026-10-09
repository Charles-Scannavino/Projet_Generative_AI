# Projet Generative AI — Assistant HLM RAG & Agent

Projet de groupe du cours **Generative AI I** : construction d'un assistant sur le logement social français combinant **RAG**, **recherche hybride**, **reranking** et **outils déterministes** sur données ouvertes.

## Objectif

L'assistant doit pouvoir :

- expliquer les droits et démarches liés au logement social ;
- répondre à partir de fiches Service-Public et, à terme, de textes LEGI ;
- vérifier l'éligibilité via des règles et barèmes calculés en Python ;
- interroger des statistiques communales RPLS ;
- calculer des indicateurs DVF sur les prix de vente ;
- citer ses sources ;
- savoir s'abstenir lorsque les données ne permettent pas de répondre.

## Contraintes du projet

- corpus cible : environ **500 documents** cohérents ;
- retrieval : **BM25 + dense**, puis **reranking** ;
- réponses structurées et validées ;
- exécution locale finale avec un modèle open-weight ;
- suivi des tokens, de la latence, des appels d'outils et du coût ;
- évaluation sur un **golden set d'environ 40 questions** ;
- commande finale attendue :

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

## État actuel

- corpus Service-Public brut chargé et parsé ;
- **3 004** fiches `F*.xml` identifiées ;
- **350** fiches du thème `Logement` retenues comme corpus Service-Public de travail ;
- filtre HLM strict : **24** fiches ;
- filtre logement élargi : **505** fiches, avec faux positifs ;
- golden set actif : **20 questions** ;
- prochaine étape : extraction et contrôle du texte complet des 350 fiches, puis ajout de LEGI et expérimentation du chunking.

Le suivi détaillé se trouve dans [Docs/AVANCEMENT.md](Docs/AVANCEMENT.md).  
Les règles et décisions durables du projet se trouvent dans [Docs/CONTEXT.md](Docs/CONTEXT.md).

## Arborescence

```text
Projet_Generative_AI/
├── README.md
├── .gitignore
├── Docs/
│   ├── CONTEXT.md
│   ├── AVANCEMENT.md
│   └── spec/
├── notebooks/
│   └── Assistant_HLM_RAG.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── indexes/
├── eval/
│   ├── golden.jsonl
│   └── golden_backlog_pending.jsonl
├── src/
└── app/
```

## Données

### Service-Public

Le fichier source actuellement utilisé est :

```text
data/raw/service_public/vosdroits-latest.zip
```

Dans Colab, les milliers de XML doivent être extraits de préférence dans `/content/` pour éviter les écritures lentes vers Google Drive.

### Sources à intégrer

- LEGI — Code de la construction et de l'habitation ;
- RPLS ;
- DVF ;
- plafonds PLAI / PLUS / PLS.

## Emplacement local obligatoire

Pour garantir le bon fonctionnement du notebook sans modification du chemin `PROJECT_PATH`, le dépôt doit être cloné ou déplacé dans l'emplacement suivant sur chaque ordinateur :

```text
~/Documents/Projets/Projet_Generative_AI
```

Le notebook utilise volontairement le chemin relatif au dossier personnel :

```python
PROJECT_PATH = "~/Documents/Projets/Projet_Generative_AI"
```

Tous les membres du groupe doivent donc conserver exactement cette arborescence locale. Le symbole `~` désigne le dossier personnel de l'utilisateur ; seul le nom d'utilisateur change d'un ordinateur à l'autre.

Pour préparer l'emplacement avant le clonage :

```bash
mkdir -p ~/Documents/Projets
cd ~/Documents/Projets
git clone https://github.com/Charles-Scannavino/Projet_Generative_AI.git
cd Projet_Generative_AI
```

## Notebook

Le notebook de prototypage est :

```text
notebooks/Assistant_HLM_RAG.ipynb
```

Le notebook sert à explorer et expérimenter. Les composants stabilisés seront ensuite déplacés dans des modules Python sous `src/`.

## Évaluation

`eval/golden.jsonl` contient actuellement 20 questions actives :

- 18 questions RAG ;
- 1 question hors périmètre ;
- 1 prompt injection.

Les questions dépendant de données non encore intégrées sont conservées dans :

```text
eval/golden_backlog_pending.jsonl
```

## Sécurité et versionnement

- ne jamais committer de clé API ;
- utiliser un fichier .env local et .gitignore ;
- ne pas committer les XML extraits, caches de modèles, index vectoriels ou résultats volumineux ;
- les versions exactes des modèles seront ajoutées au README lorsqu'elles seront figées.

## Échéances

- **Code freeze** : 20 octobre 2026 à 23 h 59 — tag `v1.0`
- **Soutenance** : 21 octobre 2026 — 10 min de pitch + 5 min de questions
