import os
import random
from typing import List

import gradio as gr

from src.database import load_dataset, load_final_summaries, load_questions
from src.rag_pipeline.rag_system import RAGSystem

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ChatInterface:
    """
    A class to create and manage the chat interface for the Dikoka AI assistant.
    """

    def __init__(self, rag_system: RAGSystem):
        """
        Initialize the ChatInterface with a RAG system.
        """
        self.rag_system = rag_system
        self.history_depth = int(os.getenv("MAX_MESSAGES") or 5) * 2
        self.questions = []
        self.summaries = []

    def respond(self, message: str, history: List[List[str]]):
        """
        Generate a response to the user's message using the RAG system.
        """
        result = ""
        history = [
            (turn["role"], turn["content"]) for turn in history[-self.history_depth :]
        ]
        for text in self.rag_system.query(message, history):
            result += text
            yield result
        return result

    def sample_questions(self):
        """
        Sample a few random questions from the loaded questions.
        """
        random_questions = random.sample(self.questions, 3)
        example_questions = "\n".join(
            ["## Examples of questions"]
            + [f"- {question}" for question in random_questions]
        )
        return example_questions

    def sample_summaries(self):
        """
        Sample a random summary from the loaded summaries.
        """
        random_summary = random.choice(self.summaries)
        return random_summary

    def load_data(self, lang: str):
        """
        Load questions and summaries for the specified language.
        """
        self.questions = load_questions(lang)
        self.summaries = load_final_summaries(lang)

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface for the chat application.
        """
        self.load_data("fr")

        description = (
            "Dikoka an AI assistant providing information on the Franco-Cameroonian Commission's"
            " findings regarding France's role and engagement in Cameroon during the suppression"
            " of independence and opposition movements between 1945 and 1971.\n\n"
            "🌟 **Code Repository**: [Dikoka GitHub](https://github.com/Nganga-AI/dikoka)"
        )

        with gr.Blocks() as demo:
            with gr.Row(equal_height=True):
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            dpd = gr.Dropdown(
                                choices=["fr", "eng"],
                                value="fr",
                                label="Choose language",
                            )
                            dpd.change(self.load_data, inputs=dpd)
                        with gr.Column(scale=2):
                            gr.Markdown("## Summary")
                    with gr.Row():
                        with gr.Column():
                            self.sample_resume = gr.Markdown(self.sample_summaries())
                    with gr.Row():
                        sample_summary = gr.Button("Sample Summary")
                        sample_summary.click(
                            fn=self.sample_summaries,
                            inputs=[],
                            outputs=self.sample_resume,
                        )
                with gr.Column(scale=2):
                    gr.ChatInterface(
                        fn=self.respond,
                        type="messages",
                        title="Dikoka",
                        description=description,
                    )
            with gr.Row():
                self.example_questions = gr.Markdown(self.sample_questions())
            with gr.Row():
                sample_button = gr.Button("Sample New Questions")
                sample_button.click(
                    fn=self.sample_questions,
                    inputs=[],
                    outputs=self.example_questions,
                )
        return demo


def get_rag_system(top_k_documents):
    """
    Initialize and return a RAG system with the specified number of top documents.
    """
    rag = RAGSystem("data/chroma_db", batch_size=64, top_k_documents=top_k_documents)
    if not os.path.exists(rag.vector_store_management.persist_directory):
        documents = load_dataset(os.getenv("LANG"))
        rag.initialize_vector_store(documents)
    return rag


# Usage example:
if __name__ == "__main__":
    top_k_docs = int(os.getenv("N_CONTEXT") or 5)
    rag_system = get_rag_system(top_k_documents=top_k_docs)

    chat_interface = ChatInterface(rag_system)
    demo = chat_interface.create_interface()
    demo.launch(share=False)
