import os
from glob import glob

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
            os.path.dirname(path) + "/" + os.path.basename(path)
        )
        for path in files
    ]
    return summaries


def load_pages_from_folder(folder_path: str, min_chunk_size=800, chunk_overlap=100):
    files = sorted(glob(os.path.join(folder_path, "*.txt")))
    name = os.path.basename(folder_path)
    pages = [open(path).read().strip() for path in files]
    content = "\n\n".join(pages)
    encoding = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=min_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoding.encode(x)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    summaries = text_splitter.split_text(content)
    return [(i, name) for i in summaries]


def load_pages_from_folders(folders_path: list[str], min_chunk_size=800, chunk_overlap=100):
    data = sum(
        [
            load_pages_from_folder(folder, min_chunk_size, chunk_overlap)
            for folder in folders_path
        ],
        start=[],
    )
    return data
