import os
from typing import Dict, List

from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_embedding
from .document_loader import DocumentLoader


def get_collection_name() -> str:
    """
    Derives the collection name from an environment variable.

    Returns:
        str: Processed collection name.
    """
    return os.getenv("HF_MODEL", "default_model").split(":")[0].split("/")[-1]


class VectorStoreManager:
    """
    Manages vector store initialization, updates, and retrieval.
    """

    def __init__(self, persist_directory: str, batch_size: int = 64):
        """
        Initializes the VectorStoreManager with the given parameters.

        Args:
            persist_directory (str): Directory to persist the vector store.
            batch_size (int): Number of documents to process in each batch.
        """
        self.persist_directory = persist_directory
        self.batch_size = batch_size
        self.embeddings = get_llm_model_embedding()
        self.collection_name = get_collection_name()
        self.vector_stores: Dict[str, Chroma] = {"chroma": None}
        self.vs_initialized = False

    def _batch_process_documents(self, documents: List[Document]):
        """
        Processes documents in batches for vector store initialization.

        Args:
            documents (List[Document]): List of documents to process.
        """
        for i in tqdm(
            range(0, len(documents), self.batch_size), desc="Processing documents"
        ):
            batch = documents[i : i + self.batch_size]
            if not self.vs_initialized:
                self.vector_stores["chroma"] = Chroma.from_documents(
                    collection_name=self.collection_name,
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
                self.vs_initialized = True
            else:
                self.vector_stores["chroma"].add_documents(batch)

    def initialize_vector_store(self, documents: List[Document] = None):
        """
        Initializes or loads the vector store.

        Args:
            documents (List[Document], optional): List of documents to initialize the vector store with.
        """
        if documents:
            self._batch_process_documents(documents)
        else:
            self.vector_stores["chroma"] = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        self.vs_initialized = True

    def create_retriever(
        self, llm, n_documents: int, bm25_portion: float = 0.8
    ) -> MultiQueryRetriever:
        """
        Creates a retriever using Chroma.

        Args:
            llm: Language model to use for the retriever.
            n_documents (int): Number of documents to retrieve.
            bm25_portion (float): Portion of BM25 to use in the retriever.

        Returns:
            MultiQueryRetriever: Configured retriever.
        """
        self.vector_store = MultiQueryRetriever.from_llm(
            retriever=self.vector_stores["chroma"].as_retriever(
                search_kwargs={"k": n_documents}
            ),
            llm=llm,
            include_original=True,
        )
        return self.vector_store

    def load_and_process_documents(self) -> List[Document]:
        """
        Loads and processes documents from the specified directory.

        Returns:
            List[Document]: List of processed documents.
        """
        loader = DocumentLoader()
        return loader.load_documents()
