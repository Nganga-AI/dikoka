import json
import random
from glob import glob

from .vector_store.document_loader import load_dataset  # noqa


def load_questions(language="fr"):
    raw: dict[str, list[dict[str, str]]] = json.load(
        open(f"saved_summaries/question_{language}.json")
    )
    questions = [example["query"] for _, fqa in raw.items() for example in fqa]
    return questions


def load_final_summaries(language="fr"):
    files = glob(f"saved_summaries/{language}/*.txt")
    data = [open(file).read() for file in files]
    random.shuffle(data)
    return data
