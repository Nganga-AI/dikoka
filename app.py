import os
from typing import List

import gradio as gr

from src.database import get_rag_system
from medivocate.src.rag_pipeline.rag_system import RAGSystem

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.history_depth = int(os.getenv("MAX_MESSAGES") or 5) * 2

    def respond(self, message: str, history: List[List[str]]):
        result = ""
        history = [(turn["role"], turn["content"]) for turn in history[-self.history_depth:]]
        for text in self.rag_system.query(message, history):
            result += text
            yield result
        return result

    def create_interface(self) -> gr.ChatInterface:
        description = (
            "Medivocate is an application that offers clear and structured information "
            "about African history and traditional medicine. The knowledge is exclusively "
            "based on historical documentaries about the African continent.\n\n"
            "🌟 **Code Repository**: [Medivocate GitHub](https://github.com/KameniAlexNea/medivocate)"
        )
        return gr.ChatInterface(
            fn=self.respond,
            type="messages",
            title="Medivocate",
            description=description,
        )


# Usage example:
if __name__ == "__main__":
    top_k_docs = int(os.getenv("N_CONTEXT") or 5)
    rag_system = get_rag_system(top_k_documents=top_k_docs)

    chat_interface = ChatInterface(rag_system)
    demo = chat_interface.create_interface()
    demo.launch(share=False)
