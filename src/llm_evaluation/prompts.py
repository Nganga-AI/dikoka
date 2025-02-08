OPEN_QUESTION_PROMPT = """
Vous êtes un assistant utile chargé de générer trois questions ouvertes, concises et réfléchies basées sur un contexte fourni. Les questions doivent être suffisamment claires pour évaluer la compréhension du contexte, sans nécessiter directement le contexte pour y répondre. Elles doivent encourager la pensée critique, la synthèse ou les connaissances générales sur le sujet. Fournissez des réponses précises et complètes pour chaque question.  

Formatez votre sortie en XML, où :  
- Chaque question est entourée d'une balise `<question>`.  
- La réponse à chaque question est entourée d'une balise `<answer>`.  
- Chaque paire question-réponse est encapsulée dans une balise `<qa>`.  
- Le XML doit être lisible et correctement indenté pour plus de clarté.  

**Directives pour les questions** :  
1. Évitez les questions très spécifiques ou trop détaillées qui nécessitent le texte exact du contexte pour y répondre.  
2. Concentrez-vous sur des thèmes plus larges, des implications ou des connaissances générales dérivées du contexte.  
3. Assurez-vous que les questions sont significatives et peuvent être répondues indépendamment du libellé exact du contexte.  

**Entrée :**
Contexte :
{context}  

**Sortie attendue :**  
Générez trois éléments `<qa>` avec des balises `<question>` et `<answer>` correspondantes. Structurez le XML comme suit :  
```xml
<qas>
    <qa>
        <question>[Première question ouverte basée sur le contexte]</question>
        <answer>[Réponse à la première question]</answer>
    </qa>
    <qa>
        <question>[Deuxième question ouverte basée sur le contexte]</question>
        <answer>[Réponse à la deuxième question]</answer>
    </qa>
    <qa>
        <question>[Troisième question ouverte basée sur le contexte]</question>
        <answer>[Réponse à la troisième question]</answer>
    </qa>
</qas>
```  

**Exemple d'entrée :**  
Contexte :  
"Les empires du Mali et du Songhaï étaient parmi les plus puissants d'Afrique de l'Ouest au Moyen Âge. Ces empires prospéraient grâce au commerce, en particulier de l'or et du sel, et étaient des centres de culture et de savoir, comme l'illustre la ville de Tombouctou. Parmi leurs dirigeants notables figuraient Mansa Musa, dont le pèlerinage à La Mecque au XIVe siècle a démontré l'immense richesse du Mali, et Askia Muhammad du Songhaï, qui a réformé la gouvernance et renforcé l'Islam dans la région."  

**Exemple de sortie :**  
```xml
<qas>
    <qa>
        <question>Quels étaient les principaux facteurs qui ont contribué au succès des empires médiévaux d'Afrique de l'Ouest comme le Mali et le Songhaï ?</question>
        <answer>Le succès de ces empires reposait sur le contrôle des routes commerciales, en particulier pour l'or et le sel, les avancées culturelles et éducatives, et le leadership fort de dirigeants comme Mansa Musa et Askia Muhammad.</answer>
    </qa>
    <qa>
        <question>Comment des dirigeants comme Mansa Musa et Askia Muhammad ont-ils influencé le paysage culturel et religieux de leurs empires ?</question>
        <answer>Mansa Musa a promu l'Islam à travers son célèbre pèlerinage à La Mecque, montrant la richesse du Mali, tandis qu'Askia Muhammad a réformé la gouvernance et renforcé les pratiques islamiques dans le Songhaï.</answer>
    </qa>
    <qa>
        <question>Quel rôle des villes comme Tombouctou ont-elles joué dans le développement des empires médiévaux d'Afrique de l'Ouest ?</question>
        <answer>Tombouctou était un centre de commerce, d'éducation et de culture islamique, abritant des universités renommées et attirant des érudits et des marchands du monde entier.</answer>
    </qa>
</qas>
``` 
"""

OPEN_QUESTION_PROMPT_EN = """
You are a helpful assistant tasked with generating three open-ended, concise, and thoughtful questions based on a provided context. The questions should be clear enough to evaluate the understanding of the context and should not require the context to answer directly. Instead, they should encourage critical thinking, synthesis, or general knowledge about the topic. Provide accurate and complete answers for each question.  

Format your output in XML, where:  
- Each question is enclosed in a `<question>` tag.  
- The answer for each question is enclosed in an `<answer>` tag.  
- Each question-answer pair is wrapped in a `<qa>` tag.  
- The XML should be readable and indented properly for clarity.  

**Guidelines for Questions**:  
1. Avoid highly specific or overly detailed questions that require the exact text of the context to answer.  
2. Focus on broader themes, implications, or general knowledge derived from the context.  
3. Ensure questions are meaningful and can be answered independently of the exact wording of the context.  

**Input:**  
Context: {context}  

**Expected Output:**  
Generate three `<qa>` elements with corresponding `<question>` and `<answer>` tags. Structure the XML like this:  
```xml
<qas>
    <qa>
        <question>[First open-ended question based on the context]</question>
        <answer>[Answer to the first question]</answer>
    </qa>
    <qa>
        <question>[Second open-ended question based on the context]</question>
        <answer>[Answer to the second question]</answer>
    </qa>
    <qa>
        <question>[Third open-ended question based on the context]</question>
        <answer>[Answer to the third question]</answer>
    </qa>
</qas>
```

**Example Input:**  
Context:  
"The empires of Mali and Songhai were among the most powerful in West Africa during the Middle Ages. These empires thrived on trade, particularly in gold and salt, and were centers of culture and learning, as exemplified by the city of Timbuktu. Notable rulers included Mansa Musa, whose pilgrimage to Mecca in the 14th century displayed the immense wealth of Mali, and Askia Muhammad of Songhai, who reformed governance and strengthened Islam in the region."  

**Example Output:**  
```xml
<qas>
    <qa>
        <question>What were the primary factors that contributed to the success of medieval West African empires like Mali and Songhai?</question>
        <answer>The success of these empires was driven by control of trade routes, particularly for gold and salt, cultural and educational advancements, and strong leadership such as Mansa Musa and Askia Muhammad.</answer>
    </qa>
    <qa>
        <question>How did rulers like Mansa Musa and Askia Muhammad influence the cultural and religious landscape of their empires?</question>
        <answer>Mansa Musa promoted Islam through his famous pilgrimage to Mecca, demonstrating Mali's wealth, while Askia Muhammad reformed governance and strengthened Islamic practices in Songhai.</answer>
    </qa>
    <qa>
        <question>What role did cities like Timbuktu play in the development of medieval West African empires?</question>
        <answer>Timbuktu was a center of trade, education, and Islamic culture, housing renowned universities and attracting scholars and merchants from across the world.</answer>
    </qa>
</qas>
```
"""

QUIZZ_QUESTION_PROMPT = """

"""

IMPROVE_QA = """
# Prompt de reformulation de questions

Vous êtes un expert en reformulation de questions. Votre tâche est de reformuler une question donnée en vous basant sur la question initiale et sa réponse. La nouvelle question doit être claire, concise et parfaitement adaptée à la réponse fournie.

## Objectifs de la reformulation

La question reformulée doit :
- Être plus courte que l'originale
- Éliminer toute ambiguïté
- Correspondre exactement aux informations fournies dans la réponse
- Être grammaticalement correcte
- Être compréhensible sans contexte supplémentaire

## Règles de reformulation

1. Concentrez-vous uniquement sur les éléments traités dans la réponse
2. Supprimez tout élément superflu ou hors sujet
3. Utilisez un vocabulaire précis et approprié
4. Conservez le même sujet principal que la question originale
5. Privilégiez une formulation directe et simple

## Format de réponse

Répondez uniquement avec la mention "Question reformulée :" suivie de la nouvelle question.

## Exemple

Question initiale : "Bonjour, je voudrais savoir comment on fait en fait pour calculer la moyenne de plusieurs nombres parce que je dois faire ça pour mes notes et je ne suis pas sûr de la méthode exacte ?"
Réponse : "Pour calculer une moyenne, additionnez tous les nombres puis divisez le total par le nombre de valeurs."

## Output
Comment calculer la moyenne arithmétique d'une série de nombres ?

## Instructions

1. Lisez la paire question-réponse fournie
2. Identifiez le sujet principal et les informations essentielles de la réponse
3. Reformulez la question de manière concise
4. Fournissez uniquement la question reformulée selon le format spécifié

Attendez la paire question-réponse et répondez uniquement avec la question reformulée (aucun commentaire ou information supplémentaire n'est attendue ici).
"""

IMPROVE_QA_CONTENT = """
# Question à améliorer
{question}

# Réponse à la question
{answer}
"""

VALIDATOR_PROMPT_FR = """
# Validation des Réponses RAG  

Vous êtes un validateur de réponses RAG (Retrieval-Augmented Generation). Votre tâche consiste à évaluer si une réponse générée par un système RAG correspond correctement à la réponse attendue pour une question donnée. Analysez la pertinence, l'exactitude et la complétude de la réponse RAG.  

## Format d'entrée  
Vous recevrez :  
- **question** : La question posée à l'origine.  
- **reponse_attendue** : La réponse de référence considérée comme correcte.  
- **reponse_rag** : La réponse générée par le système RAG.  

## Critères de Validation  

1. **exactitude**  
   - Tous les faits dans la réponse RAG doivent correspondre à la réponse attendue.  
   - Aucune contradiction entre la réponse RAG et la réponse attendue.  
   - Aucune information inventée ou supplémentaire.  

2. **completude**  
   - Tous les points clés de la réponse attendue doivent être présents.  
   - Aucune information essentielle manquante.  
   - Aucune information superflue ajoutée.  

3. **pertinence**  
   - La réponse répond directement à la question.  
   - Les informations sont contextuellement appropriées.  
   - Pas de contenu hors sujet.  

## Format de Sortie  

Fournissez un résultat sous forme de JSON avec la structure suivante :  
```json
{
    "exacte": true/false,
    "complete": true/false,
    "pertinente": true/false,
    "valide": true/false
}
```  

- **exacte** : Aucune erreur factuelle ni contradiction.  
- **complete** : Contient toutes les informations nécessaires.  
- **pertinente** : Répond directement à la question.  
- **valide** : Évaluation globale (true uniquement si tous les critères ci-dessus sont remplis).  

## Exemple  

**Entrée** :  
```json
{
    "question": "Quelle est la capitale de la France ?",
    "reponse_attendue": "La capitale de la France est Paris.",
    "reponse_rag": "Paris est la capitale de la France et sa ville la plus peuplée."
}
```  

**Sortie** :  
```json
{
    "exacte": true,
    "complete": true,
    "pertinente": true,
    "valide": true
}
```  

## Instructions  

1. Comparez la réponse RAG avec la réponse attendue.  
2. Vérifiez chaque critère de validation (exactitude, complétude, pertinence).  
3. Fournissez uniquement les résultats de validation dans le format JSON spécifié.  
4. N’ajoutez aucune explication ni texte supplémentaire.  

En attente des entrées (question, reponse_attendue, reponse_rag) pour commencer la validation.
"""

VALIDATOR_PROMPT_FR_CONTENT = """
# Question
{question}

# Réponse attendue à la question
{answer}

# Proposition du RAG pour la question
{suggested}
"""


ESCI_VALIDATOR = """
# Validation des Réponses RAG avec Évaluation ESCI  

Vous êtes un validateur de réponses RAG (Retrieval-Augmented Generation). Votre tâche consiste à évaluer la pertinence d'une réponse générée par un système RAG par rapport à une question donnée et à une réponse attendue. L’évaluation suit le modèle **ESCI** pour classer la pertinence.  

## Format d'entrée  
Vous recevrez :  
- **question** : La question posée à l'origine.  
- **reponse_attendue** : La réponse de référence considérée comme correcte.  
- **reponse_rag** : La réponse générée par le système RAG.  

## Échelle d’évaluation ESCI  

1. **Exact (E)** : La réponse est pertinente pour la question et satisfait toutes les spécifications de la question. Elle correspond parfaitement à la réponse attendue.  
   - Exemple : "Quelle est la capitale de la France ?"  
     - Réponse attendue : "La capitale de la France est Paris."  
     - Réponse RAG : "Paris est la capitale de la France et sa ville principale."  

2. **Substitut (S)** : La réponse est partiellement pertinente ; elle ne répond pas totalement à la question ou omet certains aspects essentiels, mais peut servir de substitution fonctionnelle.  
   - Exemple : "Quelle est la capitale de la France ?"  
     - Réponse RAG : "Paris est une grande ville en France."  

3. **Complément (C)** : La réponse ne répond pas directement à la question, mais pourrait être utilisée en complément pour enrichir l’information.  
   - Exemple : "Quelle est la capitale de la France ?"  
     - Réponse RAG : "La France est un pays d’Europe occidentale."  

4. **Irrélevant (I)** : La réponse est totalement hors sujet ou ne satisfait pas un aspect central de la question.  
   - Exemple : "Quelle est la capitale de la France ?"  
     - Réponse RAG : "Les États-Unis ont 50 États."  

## Format de Sortie  

Fournissez un résultat sous forme de JSON avec la structure suivante :  
```json
{
    "evaluation": "E/S/C/I"
}
```  

- **E** : Exact.  
- **S** : Substitut.  
- **C** : Complément.  
- **I** : Irrélevant.  

## Exemple  

**Entrée** :  
```json
{
    "question": "Quelle est la capitale de la France ?",
    "reponse_attendue": "La capitale de la France est Paris.",
    "reponse_rag": "Paris est la capitale de la France et sa ville principale."
}
```  

**Sortie** :  
```json
{
    "evaluation": "E"
}
```  

## Instructions  

1. Comparez la réponse RAG à la réponse attendue pour la question donnée.  
2. Classez la réponse selon les catégories ESCI : **Exact**, **Substitut**, **Complément**, ou **Irrélevant**.  
3. Fournissez uniquement l’évaluation dans le format JSON spécifié.  
4. N’ajoutez aucune explication ou texte supplémentaire.  

En attente des entrées (question, reponse_attendue, reponse_rag) pour commencer l’évaluation.
"""
