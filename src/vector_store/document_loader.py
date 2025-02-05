import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from tqdm import tqdm


def sanitize_metadata(metadata: dict) -> dict:
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            sanitized[key] = ", ".join(value)
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            raise ValueError(
                f"Unsupported metadata type for key '{key}': {type(value)}"
            )
    return sanitized


class DocumentLoader:
    """
    Handles loading and splitting documents from directories.
    """

    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir

    def load_text_documents(self) -> List[Document]:
        """Loads and splits text documents."""
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)

    def load_json_documents(self) -> List[Document]:
        """Loads and processes JSON documents."""
        files = glob(os.path.join(self.docs_dir, "*.json"))

        def load_json_file(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)["kwargs"]
            return Document.model_validate(
                {**data, "metadata": sanitize_metadata(data["metadata"])}
            )

        with ThreadPoolExecutor() as executor:
            documents = list(
                tqdm(
                    executor.map(load_json_file, files),
                    total=len(files),
                    desc="Loading JSON documents",
                )
            )

        return documents

    def load_documents(self) -> List[Document]:
        """Determines and loads documents based on file type."""
        if glob(os.path.join(self.docs_dir, "*.json")):
            return self.load_json_documents()
        return self.load_text_documents()
