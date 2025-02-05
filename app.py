import os
import random
from typing import List

import gradio as gr

from medivocate.src.rag_pipeline.rag_system import RAGSystem
from src.database import get_rag_system, load_final_summaries, load_questions

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.history_depth = int(os.getenv("MAX_MESSAGES") or 5) * 2

    def respond(self, message: str, history: List[List[str]]):
        result = ""
        history = [
            (turn["role"], turn["content"]) for turn in history[-self.history_depth :]
        ]
        for text in self.rag_system.query(message, history):
            result += text
            yield result
        return result

    def sample_questions(self, questions: List[str]):
        random_questions = random.sample(questions, 3)
        example_questions = "\n".join(
            ["### Examples of questions"]
            + [f"- {question}" for question in random_questions]
        )
        return example_questions

    def sample_summaries(self, summaries: list[str]):
        random_summary = random.choice(summaries)
        return f"### Summary\n{random_summary}"

    def create_interface(self) -> gr.Blocks:
        questions: list[str] = load_questions()
        summaries: list[str] = load_final_summaries()
        description = (
            "Dikoka an AI assistant providing information on the Franco-Cameroonian Commission's"
            " findings regarding France's role and engagement in Cameroon during the suppression"
            " of independence and opposition movements between 1945 and 1971.\n\n"
            "🌟 **Code Repository**: [Dikoka GitHub](https://github.com/Nganga-AI/dikoka)"
        )

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        random_summary = random.choice(summaries)
                        sample_resume = gr.Markdown(f"### Summary\n{random_summary}")
                    with gr.Row():
                        sample_summary = gr.Button("Sample Summary")
                        sample_summary.click(
                            fn=lambda: self.sample_summaries(summaries),
                            inputs=[],
                            outputs=sample_resume,
                        )
                with gr.Column(scale=2):
                    with gr.Row():
                        gr.ChatInterface(
                            fn=self.respond,
                            type="messages",
                            title="Dikoka",
                            description=description,
                        )
            with gr.Row():
                example_questions = gr.Markdown(self.sample_questions(questions))
            with gr.Row():
                sample_button = gr.Button("Sample New Questions")
                sample_button.click(
                    fn=lambda: self.sample_questions(questions),
                    inputs=[],
                    outputs=example_questions,
                )
        return demo


# Usage example:
if __name__ == "__main__":
    top_k_docs = int(os.getenv("N_CONTEXT") or 5)
    rag_system = get_rag_system(top_k_documents=top_k_docs)

    chat_interface = ChatInterface(rag_system)
    demo = chat_interface.create_interface()
    demo.launch(share=False)
