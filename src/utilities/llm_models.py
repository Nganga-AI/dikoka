import os
from enum import Enum

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


class LLMModel(Enum):
    OLLAMA = ChatOllama
    GROQ = ChatGroq


def get_llm_model_chat(temperature=0.01, max_tokens: int=None):
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
