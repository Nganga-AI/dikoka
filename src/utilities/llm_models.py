import os
from enum import Enum

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .embedding import CustomEmbedding


class LLMModel(Enum):
    OLLAMA = "ChatOllama"
    GROQ = "ChatGroq"


def get_llm_model_chat(temperature=0.01, max_tokens: int = None):
    if str(os.getenv("USE_OLLAMA_CHAT")) == "1":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL"),
            temperature=temperature,
            num_predict=max_tokens,
        )
    return ChatGroq(
        model=os.getenv("GROQ_MODEL_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm_model_embedding():
    if str(os.getenv("USE_HF_EMBEDDING")) == "1":
        return CustomEmbedding()
    return OllamaEmbeddings(
        model=os.getenv("OLLAM_EMB"),
        base_url=(
            os.getenv("OLLAMA_HOST") if os.getenv("OLLAMA_HOST") is not None else None
        ),
        client_kwargs=(
            {
                "headers": {
                    "Authorization": "Bearer " + (os.getenv("OLLAMA_TOKEN") or "")
                }
            }
            if os.getenv("OLLAMA_HOST") is not None
            else None
        ),
    )
