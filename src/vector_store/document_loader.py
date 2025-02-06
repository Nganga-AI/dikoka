import json
import os
from glob import glob
from typing import List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_qa_dataset(file_path: str):
    raw: dict[str, list[dict[str, str]]] = json.load(open(file_path))
    questions = [
        [example["response"], os.path.basename(path)]
        for path, fqa in raw.items()
        for example in fqa
    ]
    return questions


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


def load_dataset(language="fr"):
    question_path = f"saved_summaries/question_{language}.json"
    questions = load_qa_dataset(question_path)

    summary_path = "data/summaries/summaries_" + language
    summaries = load_summaries(summary_path)

    pages = load_pages_from_folders(
        ["data/pages/297054", "data/pages/297054_Volume_2"],
        chunk_size=512,
        chunk_overlap=100,
    )

    documents = questions + summaries + pages
    documents = [
        Document(page_content=doc, metadata={"source": source})
        for (doc, source) in documents
    ]
    return documents


class DocumentLoader:
    """
    Handles loading and splitting documents from directories.
    """

    def load_documents(self) -> List[Document]:
        """Determines and loads documents based on file type."""
        return load_dataset()
