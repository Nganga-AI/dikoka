import json
import random
from glob import glob

from .vector_store.document_loader import load_dataset  # noqa


def load_questions():
    qa_ds = json.load(open("saved_summaries/questions.json"))
    return qa_ds


def load_final_summaries():
    files = glob("saved_summaries/*.txt") + glob("saved_summaries/*.txt")
    data = [open(file).read() for file in files]
    random.shuffle(data)
    return data
