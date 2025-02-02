import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import pymupdf
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    content: str
    page_number: int


def get_page_data(page: pymupdf.Page, page_id):
    return PDFPage(pymupdf.utils.get_text(page), page_id)


class PDFReader:

    def pdf_to_texts_batch(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        batch_size: int = 8,
    ):
        pdf_path = Path(pdf_path)

        doc = pymupdf.open(pdf_path)
        if pages is None:
            pages = list(range(len(doc)))
        page_batched = [
            pages[i : i + batch_size] for i in range(0, len(pages), batch_size)
        ]

        with ThreadPoolExecutor() as mapper:
            documents = list(
                tqdm(
                    mapper.map(
                        lambda batch: [get_page_data(doc[i], i) for i in batch],
                        page_batched,
                    ),
                    total=len(page_batched),
                )
            )
        return documents

    def convert_document_to_text(
        self,
        path: Union[str, Path],
        pages: Optional[List[int]] = None,
        batch_size: int = 8,
        output_folder: str = None,
    ):
        documents = [
            page
            for batch in self.pdf_to_texts_batch(path, pages, batch_size)
            for page in batch
        ]
        # documents = sorted(documents, key=lambda x: x.page_number)
        output_folder = (
            str(path).replace(".pdf", "") if output_folder is None else output_folder
        )
        os.makedirs(output_folder, exist_ok=True)
        for document in documents:
            if len(document.content.strip()) > 10:
                with open(
                    os.path.join(output_folder, f"page_{document.page_number}.txt"), "w"
                ) as file:
                    file.write(document.content)

    def convert_documents_to_text(
        self,
        folder: Union[str, Path],
        batch_size: int = 8,
    ):
        paths = glob(os.path.join(folder, "*.pdf")) + glob(
            os.path.join(folder, "*/*.pdf")
        )
        for path in tqdm(paths):
            self.convert_document_to_text(path, batch_size=batch_size)


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument("--pdf_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=8, required=False)
    args = args.parse_args()
    reader = PDFReader()
    reader.convert_documents_to_text(args.pdf_path, args.batch_size)
