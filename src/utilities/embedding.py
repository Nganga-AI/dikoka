import logging
import os
from typing import Any, List

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
)
from pydantic import BaseModel, Field


class CustomEmbedding(BaseModel, Embeddings):
    """
    Custom embedding class that supports both hosted and CPU embeddings.
    """

    hosted_embedding: HuggingFaceEndpointEmbeddings = Field(
        default_factory=lambda: None
    )
    cpu_embedding: HuggingFaceEmbeddings = Field(default_factory=lambda: None)
    matryoshka_dim: int = Field(default=256)

    def get_instruction(self) -> str:
        """
        Generates the instruction for the embedding model based on environment variables.

        Returns:
            str: The instruction string.
        """
        if "nomic" in os.getenv("HF_MODEL"):
            return (
                "search_query: "
                if (os.getenv("IS_APP", "0") == "1")
                else "search_document: "
            )
        return (
            "Represent this sentence for searching relevant passages:"
            if (os.getenv("IS_APP", "0") == "1")
            else ""
        )

    def get_hf_embedd(self) -> HuggingFaceEmbeddings:
        """
        Initializes the HuggingFaceEmbeddings with the appropriate settings.

        Returns:
            HuggingFaceEmbeddings: The initialized HuggingFaceEmbeddings object.
        """
        return HuggingFaceEmbeddings(
            model_name=os.getenv("HF_MODEL"),  # You can replace with any HF model
            model_kwargs={
                "device": "cpu" if not torch.cuda.is_available() else "cuda",
                "trust_remote_code": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "prompt": self.get_instruction(),
            },
        )

    def __init__(self, matryoshka_dim=256, **kwargs: Any):
        """
        Initializes the CustomEmbedding with the given parameters.

        Args:
            matryoshka_dim (int): Dimension of the embeddings.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        query_instruction = self.get_instruction()
        self.matryoshka_dim = matryoshka_dim
        if torch.cuda.is_available():
            logging.info("CUDA is available")
            self.hosted_embedding = self.get_hf_embedd()
            self.cpu_embedding = self.hosted_embedding
        else:
            logging.info("CUDA is not available")
            self.hosted_embedding = HuggingFaceEndpointEmbeddings(
                model=os.getenv("HF_MODEL"),
                model_kwargs={
                    "encode_kwargs": {
                        "normalize_embeddings": True,
                        "prompt": query_instruction,
                    }
                },
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            )
            self.cpu_embedding = self.get_hf_embedd()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the appropriate embedding model.

        Args:
            texts (List[str]): List of document texts to embed.

        Returns:
            List[List[float]]: List of embedded document vectors.
        """
        try:
            embed = self.hosted_embedding.embed_documents(texts)
        except Exception as e:
            logging.warning(f"Issue with batch hosted embedding, moving to CPU: {e}")
            embed = self.cpu_embedding.embed_documents(texts)
        return (
            [e[: self.matryoshka_dim] for e in embed] if self.matryoshka_dim else embed
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query using the appropriate embedding model.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: The embedded query vector.
        """
        try:
            embed = self.hosted_embedding.embed_query(text)
        except Exception as e:
            logging.warning(f"Issue with hosted embedding, moving to CPU: {e}")
            embed = self.cpu_embedding.embed_query(text)
        logging.warning(text)
        return embed[: self.matryoshka_dim] if self.matryoshka_dim else embed
