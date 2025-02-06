import argparse
import logging
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_chat
from .prompts import TRANSLATE_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslateSummary:
    def __init__(
        self,
        output_folder: str = "summaries",
    ):
        # Setup tokenizer and parameters
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        # Initialize LLM
        self.llm = get_llm_model_chat(
            temperature=0.8,
            max_tokens=2500,
        )
        self.translate_prompt = TRANSLATE_PROMPT

    def read_documents(self, folder_path: str) -> List[str]:
        """Read and return contents of all .txt files in the folder (sorted)."""
        folder = Path(folder_path)
        txt_files = sorted(folder.glob("*.txt"))
        documents = []
        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    documents.append([file.read(), file_path])
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
        return documents

    def translate_chunk(self, chunk_text: str) -> str:
        """Summarize a single chunk using the summary prompt."""
        summary_chain = self.translate_prompt | self.llm
        result = summary_chain.invoke({"text": chunk_text})
        return result.content.strip()

    def translate(self, folder: str):
        files = self.read_documents(folder)
        for dat_file, file in tqdm(files):
            tr = self.translate_chunk(dat_file)
            with open(
                os.path.join(self.output_folder, os.path.basename(file)), "w"
            ) as f:
                f.write(tr)


def main(
    folder_path: str = "data/summaries/297054", output_folder: str = "data/summaries_fr"
):
    output_folder = os.path.join(output_folder, os.path.basename(folder_path))
    os.makedirs(output_folder, exist_ok=True)
    summarizer = TranslateSummary(
        output_folder=output_folder,
    )
    summarizer.translate(folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Summarizer")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="data/summaries/297054",
        help="Path to the folder containing text files to summarize",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/summaries_fr",
        help="Path to the folder to save summaries",
    )
    args = parser.parse_args()
    main(folder_path=args.folder_path, output_folder=args.output_folder)
