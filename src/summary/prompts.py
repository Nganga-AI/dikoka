from langchain.prompts import PromptTemplate

SUMMARY_PROMPT = PromptTemplate(
    template="""You are tasked with creating an iterative summary of a long document. Your goal is to summarize the given text while preserving key information and insights. This summary will be part of a larger document summary, so it's important to maintain coherence with previous summaries.

Here is the text you need to summarize:
<text>
{text}
</text>

Here is the context from previous summaries (this may be empty if you're summarizing the first page):
<context>
{context}
</context>

Follow these steps to create your summary:

1. Read the provided text carefully.
2. If context is provided, review it to understand what has been summarized previously.
3. Identify the main ideas, key points, and essential details in the text.
4. Create a concise summary that:
   a. Captures the most important information
   b. Maintains the original meaning and tone
   c. Avoids redundancy with the context (if provided)
   d. Preserves any crucial facts, figures, or insights
   e. Ensures coherence with the previous summaries (if context is provided)
5. Keep your summary clear, objective, and well-structured.

SUMMARY:""",
    input_variables=["text", "context"],
)

FINAL_PROMPT = PromptTemplate(
    template="""You are tasked with creating a final summary based on a set of section summaries. Your goal is to synthesize the information from these summaries into a coherent, logically flowing final summary that preserves key insights from each section.

Here are the section summaries you will be working with:

<section_summaries>
{SECTION_SUMMARIES}
</section_summaries>

To create the final summary, follow these steps:

1. Carefully read through all the section summaries to understand the main topics and key points.

2. Identify the overarching themes or main ideas that connect the different sections.

3. Organize the information in a logical order, ensuring a smooth flow from one topic to another.

4. Synthesize the key insights from each section, avoiding repetition and maintaining coherence.

5. Ensure that the final summary captures the essence of all sections without favoring any particular one.

6. Use transitional phrases to connect ideas and maintain a smooth flow throughout the summary.

7. Keep the language clear, concise, and appropriate for the subject matter.

8. Aim for a final summary that is comprehensive yet concise, typically about 25 percent of the total length of all section summaries combined.

Remember to maintain coherence and logical flow while preserving key insights from each section. Your summary should provide a clear overview of the entire content without losing important details from individual sections.

FINAL SUMMARY:""",
    input_variables=["SECTION_SUMMARIES"],
)

TRANSLATE_PROMPT = PromptTemplate(
    template="""You are tasked with translating a given text from English to French. Your goal is to provide an accurate translation without adding any additional information or context. Follow these steps:

1. Here is the text you need to translate:
{text}

2. Translate the above text into French, adhering to these guidelines:
   - Maintain the original meaning and tone of the text.
   - Do not add any explanations, context, or additional information.
   - Preserve the original formatting, including line breaks and paragraph structure.
   - If there are any proper nouns or specific terms that should not be translated, keep them in their original form.
""",
    input_variables=["text"],
)

TRANSLATE_QA = PromptTemplate(
    template="""You are tasked with translating a given text from English to French. Your goal is to provide an accurate translation without adding any additional information or context. Follow these steps:

1. Here is the text you need to translate:
{text}

2. Translate the above text into French, adhering to these guidelines:
   - Maintain the original meaning and tone of the text.
   - Do not add any explanations, context, or additional information.
   - Preserve the original formatting, including line breaks and paragraph structure.
   - If there are any proper nouns or specific terms that should not be translated, keep them in their original form.
""",
    input_variables=["text"],
)