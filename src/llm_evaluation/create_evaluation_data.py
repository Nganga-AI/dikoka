"""
python -m src.llm_evaluation.create_evaluation_data --input_folder data/chunks --output_folder data/evaluation --n_files 250 --file_type json
"""

import argparse
import json
import os
import random
from glob import glob

from tqdm import tqdm
import logging

from ..utilities.llm_models import get_llm_model_chat
from .prompts import OPEN_QUESTION_PROMPT_EN as OPEN_QUESTION_PROMPT


def load_data(path: str) -> str:
    if path.endswith(".txt"):
        return open(path).read()
    return json.load(open(path))["kwargs"]["page_content"]


def generate_questions(
    input_folder: str, n_files: int, output_folder: str, file_type="json"
):
    """
    Generate questions using an LLM based on text files in a folder and save the results in a specified folder.

    Args:
        input_folder (str): Path to the folder containing input text files.
        n_files (int): Number of files to process.
        output_folder (str): Path to the folder where output files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = glob(os.path.join(input_folder, "*." + file_type)) + glob(
        os.path.join(input_folder, "*/*." + file_type)
    )
    logging.info(f"Found {len(files)} files in {input_folder}.")

    if len(files):
        files = random.sample(files, min(n_files, len(files)))

        llm = get_llm_model_chat(temperature=0.8, max_tokens=1500)

        for file in tqdm(files):
            lines = load_data(file)
            if len(lines.strip().split()) < 100:
                logging.warning(f"Ignoring {file} (too few words)")
                continue

            name = os.path.basename(file).replace(file_type, "")

            # Generate a unique filename for the output
            output_filename = f"{name}txt"
            output_path = os.path.join(output_folder, output_filename)
            if os.path.isfile(output_path):
                continue

            # Generate the text using the LLM
            text = llm.invoke([("user", OPEN_QUESTION_PROMPT.format(context=lines))])

            # Save the generated content to the output file
            with open(output_path, "w") as out_file:
                out_file.write(text.content)

            logging.info(f"Saved generated questions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions from text files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/summaries",
        help="Path to the folder containing input text files.",
    )
    parser.add_argument(
        "--n_files", type=int, default=250, help="Number of files to process."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/questions",
        help="Path to the folder where output files will be saved.",
    )
    parser.add_argument(
        "--file_type",
        default="txt",  # json ou txt
        type=str,
        help="Type of file to consider",
    )

    args = parser.parse_args()

    generate_questions(
        args.input_folder, args.n_files, args.output_folder, args.file_type
    )
