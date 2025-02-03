import os
from glob import glob

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .rag_pipeline.rag_system import RAGSystem

from .llm_evaluation.improve_generated_qa import (
    parse_questions_answers_with_regex,
)


def load_qa_dataset(folder_path: str):
    qa_ds = parse_questions_answers_with_regex(folder_path)
    return [(i[1], "QA_LLM") for i in qa_ds]


def load_summaries(folder_path: str):
    files = sorted(glob(os.path.join(folder_path, "*.txt"))) + sorted(
        glob(os.path.join(folder_path, "*/*.txt"))
    )
    summaries = [
        (
            open(path).read().strip(),
            os.path.dirname(path) + "/" + os.path.basename(path),
        )
        for path in files
    ]
    return summaries


def load_pages_from_folder(folder_path: str, chunk_size=512, chunk_overlap=100):
    files = sorted(glob(os.path.join(folder_path, "*.txt")))
    name = os.path.basename(folder_path)
    pages = [open(path).read().strip() for path in files]
    content = "\n\n".join(pages)
    encoding = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoding.encode(x)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    summaries = text_splitter.split_text(content)
    return [(i, name) for i in summaries]


def load_pages_from_folders(folders_path: list[str], chunk_size=512, chunk_overlap=100):
    data = sum(
        [
            load_pages_from_folder(folder, chunk_size, chunk_overlap)
            for folder in folders_path
        ],
        start=[],
    )
    return data


def load_dataset():
    questions = load_qa_dataset("data/questions")
    summaries = load_summaries("data/summaries")
    pages = load_pages_from_folders(
        ["data/297054", "data/297054_Volume_2"], chunk_size=512, chunk_overlap=100
    )
    documents = questions + summaries + pages
    documents = [
        Document(page_content=doc, metadata={"source": source})
        for (doc, source) in documents
    ]
    return documents


def get_rag_system(top_k_documents):
    rag = RAGSystem(
        None, "data/chroma_db", batch_size=64, top_k_documents=top_k_documents
    )
    if not os.path.exists(rag.vector_store_management.persist_directory):
        documents = load_dataset()
        rag.initialize_vector_store(documents)
    return rag
