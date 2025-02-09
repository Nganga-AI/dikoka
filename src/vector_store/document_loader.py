import json
import os
from glob import glob
from typing import List, Dict

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_qa_dataset(file_path: str) -> List[List[str]]:
    """
    Load question-answer dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the QA dataset.

    Returns:
        List[List[str]]: List of questions and their corresponding file names.
    """
    raw: Dict[str, List[Dict[str, str]]] = json.load(open(file_path))
    questions = [
        [example["response"], os.path.basename(path)]
        for path, fqa in raw.items()
        for example in fqa
    ]
    return questions


def load_summaries(folder_path: str, chunk_size=512, chunk_overlap=100) -> List[List[str]]:
    """
    Load summaries from text files in a folder and split them into chunks.

    Args:
        folder_path (str): Path to the folder containing text files.
        chunk_size (int, optional): Size of each chunk. Defaults to 512.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 100.

    Returns:
        List[List[str]]: List of text chunks and their corresponding folder names.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoding.encode(x)),
    )

    files = sorted(glob(os.path.join(folder_path, "**/*.txt"), recursive=True))
    grouped_files: Dict[str, List[str]] = {}
    for file in files:
        folder = os.path.dirname(file)
        grouped_files.setdefault(folder, []).append(file)

    summaries = []
    for folder, file_list in grouped_files.items():
        content = "\n\n".join(open(file).read() for file in file_list)
        summaries.extend((chunk, folder) for chunk in text_splitter.split_text(content))

    return summaries


def load_pages_from_folder(folder_path: str, chunk_size=512, chunk_overlap=100) -> List[List[str]]:
    """
    Load pages from text files in a folder and split them into chunks.

    Args:
        folder_path (str): Path to the folder containing text files.
        chunk_size (int, optional): Size of each chunk. Defaults to 512.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 100.

    Returns:
        List[List[str]]: List of text chunks and their corresponding folder names.
    """
    files = sorted(glob(os.path.join(folder_path, "*.txt")))
    name = os.path.basename(folder_path)
    pages = [open(path).read().strip() for path in files]
    content = "\n\n".join(pages)
    encoding = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(encoding.encode(x)),
    )
    summaries = text_splitter.split_text(content)
    return [(i, name) for i in summaries]


def load_pages_from_folders(folders_path: List[str], chunk_size=512, chunk_overlap=100) -> List[List[str]]:
    """
    Load pages from multiple folders and split them into chunks.

    Args:
        folders_path (List[str]): List of folder paths containing text files.
        chunk_size (int, optional): Size of each chunk. Defaults to 512.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 100.

    Returns:
        List[List[str]]: List of text chunks and their corresponding folder names.
    """
    data = sum(
        [
            load_pages_from_folder(folder, chunk_size, chunk_overlap)
            for folder in folders_path
        ],
        start=[],
    )
    return data


def load_dataset(language="fr") -> List[Document]:
    """
    Load the entire dataset including questions, summaries, and pages.

    Args:
        language (str, optional): Language of the dataset. Defaults to "fr".

    Returns:
        List[Document]: List of Document objects containing the dataset.
    """
    question_path = f"saved_summaries/question_{language}.json"
    questions = load_qa_dataset(question_path)

    summary_path = "data/summaries/summaries_" + language
    summaries = load_summaries(
        summary_path,
        chunk_size=512,
        chunk_overlap=100,
    )

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
        """
        Determines and loads documents based on file type.

        Returns:
            List[Document]: List of Document objects containing the dataset.
        """
        return load_dataset()
