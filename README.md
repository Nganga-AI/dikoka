# Dikoka

Dikoka est un analyseur de documents propulsé par l'IA, conçu pour extraire des informations clés, générer des résumés concis et suggérer des questions de suivi pour approfondir la compréhension. Le projet se focalise sur l'implication de la France au Cameroun lors de la répression des mouvements indépendantistes et d'opposition (1945-1971), d'après les conclusions de la Commission Franco-Camerounaise.

## Fonctionnalités

* **Chargement et Traitement de Documents** : Charger et traiter des documents depuis un répertoire spécifié.
* **Gestion du Vector Store** : Initialiser, mettre à jour et récupérer des documents à l'aide d'un vector store.
* **Résumé Hiérarchique** : Générer un résumé hiérarchique des documents.

  *Cette approche permet d'organiser le contenu en sections et paragraphes structurés, facilitant ainsi la compréhension globale des documents longs.*
* **Dataset de Questions** : Un ensemble de questions préétablies qui sert d'exemple pour guider l'exploration du contenu.
* **Retrieval-Augmented Generation (RAG)** : Récupérer et générer des réponses aux questions en se basant sur le contenu des documents.
* **Multi Query Splitting** : Inspiré par LangChain Chat, cette fonctionnalité segmente les questions complexes en sous-parties pour améliorer la précision des réponses.
* **Interface de Chat** : Une interface interactive permettant d'interroger l'assistant IA.

## Justification de l'Approche

Pour traiter efficacement des documents volumineux et complexes, Dikoka adopte plusieurs améliorations par rapport à une approche RAG naïve :

* **Résumé Hiérarchique** : En organisant le contenu en sections clairement définies, cette méthode aide à identifier rapidement les thèmes et les informations essentielles, améliorant ainsi la lisibilité et la compréhension globale du document.
* **Dataset de Questions** : Proposer un ensemble de questions types permet aux utilisateurs de découvrir des aspects qu’ils n’auraient pas envisagés, facilitant l’apprentissage de nouvelles informations et l’exploration approfondie du contenu.
* **Multi Query Splitting** : En divisant les questions en groupes plus petits, cette technique permet de traiter chaque segment de manière plus précise, optimisant ainsi la pertinence des réponses générées.

Ces éléments combinés rendent Dikoka particulièrement efficace pour analyser des documents historiques complexes, tout en offrant une expérience utilisateur améliorée.

## Installation

Pour installer les dépendances requises, exécutez :

```bash
pip install -r requirements.txt
```

## Utilisation

### Exécution du Résumeur Hiérarchique

Pour lancer le résumeur hiérarchique, utilisez la commande suivante :

```bash
python src/summary/summarizer.py --folder_path <chemin_du_dossier> --output_folder <chemin_du_dossier_de_sortie>
```

### Exécution du Système RAG

Pour initialiser et lancer le système RAG, utilisez la commande suivante :

```bash
python app.py
```

### Variables d'Environnement

Configurez les variables d'environnement suivantes pour paramétrer les modèles et autres réglages :

* `HF_MODEL`: Nom du modèle Hugging Face.
* `USE_OLLAMA_CHAT`: Mettre à `1` pour utiliser ChatOllama.
* `OLLAMA_MODEL`: Nom du modèle Ollama.
* `GROQ_MODEL_NAME`: Nom du modèle Groq.
* `USE_HF_EMBEDDING`: Mettre à `1` pour utiliser les embeddings Hugging Face.
* `OLLAM_EMB`: Nom du modèle d'embedding Ollama.
* `OLLAMA_HOST`: URL de l'hôte Ollama.
* `OLLAMA_TOKEN`: Jeton API d'Ollama.
* `HUGGINGFACEHUB_API_TOKEN`: Jeton API du Hugging Face Hub.
* `MAX_MESSAGES`: Nombre maximum de messages à conserver dans l'historique du chat.
* `N_CONTEXT`: Nombre de documents à récupérer dans le contexte.

## Structure du Projet

* `src/vector_store/vector_store.py` : Gère l'initialisation, la mise à jour et la récupération du vector store.
* `src/utilities/llm_models.py` : Fournit des fonctions pour obtenir les modèles de langage et les embeddings.
* `src/utilities/embedding.py` : Définit la classe d'embedding personnalisée.
* `src/summary/summarizer.py` : Implémente le résumeur hiérarchique.
* `src/rag_pipeline/rag_system.py` : Implémente le système RAG pour la récupération de documents et la réponse aux questions.
* `src/rag_pipeline/prompts.py` : Définit les prompts pour le système RAG.
* `app.py` : Script principal pour lancer l'interface de chat.

## Exemple

Pour tester Dikoka, suivez ces étapes :

1. Clonez le dépôt :
   ```bash
   git clone git@github.com:Nganga-AI/medivocate.git
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Exécutez le résumeur hiérarchique :
   ```bash
   python src/summary/summarizer.py --folder_path data/297054_Volume_2 --output_folder data/summaries
   ```
4. Lancez l'interface de chat :
   ```bash
   python app.py
   ```

## Contribuer

Les contributions sont les bienvenues ! Merci d'ouvrir une issue ou de soumettre une pull request sur GitHub.

## Licence

Ce projet est sous licence MIT.
