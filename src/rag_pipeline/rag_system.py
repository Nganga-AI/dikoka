import logging
from typing import List, Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain.chains.retrieval import create_retrieval_chain

from ..utilities.llm_models import get_llm_model_chat
from ..vector_store.vector_store import VectorStoreManager
from .prompts import CHAT_PROMPT, CONTEXTUEL_QUERY_PROMPT


class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system for document retrieval and question answering.
    """

    def __init__(
        self,
        persist_directory_dir="data/chroma_db",
        batch_size: int = 64,
        top_k_documents=5,
    ):
        """
        Initializes the RAGSystem with the given parameters.

        Args:
            persist_directory_dir (str): Directory to persist the vector store.
            batch_size (int): Number of documents to process in each batch.
            top_k_documents (int): Number of top documents to retrieve.
        """
        self.top_k_documents = top_k_documents
        self.llm = self._get_llm()
        self.chain: Optional[BaseConversationalRetrievalChain] = None
        self.vector_store_management = VectorStoreManager(
            persist_directory_dir, batch_size
        )

    def _get_llm(self):
        """
        Retrieves the language model for the RAG system.

        Returns:
            The language model.
        """
        return get_llm_model_chat(temperature=0.1, max_tokens=1000)

    def load_documents(self) -> List:
        """
        Loads and splits documents from the specified directory.

        Returns:
            List: List of loaded documents.
        """
        return self.vector_store_management.load_and_process_documents()

    def initialize_vector_store(self, documents: List = None):
        """
        Initializes or loads the vector store.

        Args:
            documents (List, optional): List of documents to initialize the vector store. Defaults to None.
        """
        self.vector_store_management.initialize_vector_store(documents)

    def setup_rag_chain(self):
        """
        Sets up the RAG chain for document retrieval and question answering.
        """
        if self.chain is not None:
            return
        retriever = self.vector_store_management.create_retriever(
            self.llm, self.top_k_documents, bm25_portion=0.03
        )

        # Contextualize question
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, CONTEXTUEL_QUERY_PROMPT
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, CHAT_PROMPT)
        self.chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        logging.info("RAG chain setup complete" + str(self.chain))
        return self.chain

    def query(self, question: str, history: list = []):
        """
        Queries the RAG system with a question and chat history.

        Args:
            question (str): The question to query.
            history (list, optional): The chat history. Defaults to [].

        Yields:
            str: The answer from the RAG system.
        """
        if not self.vector_store_management.vs_initialized:
            self.initialize_vector_store()

        self.setup_rag_chain()

        for token in self.chain.stream({"input": question, "chat_history": history}):
            if "answer" in token:
                yield token["answer"]
