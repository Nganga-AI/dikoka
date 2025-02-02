"""
python -m src.llm_evaluation.improve_generated_qa --input_folder data/evaluation --output_folder data/clear_evaluation
"""

import argparse
import json
import os
import re
import uuid
from glob import glob

from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_chat
from .prompts import IMPROVE_QA, IMPROVE_QA_CONTENT


def parse_questions_answers_with_regex_file(file):
    question_pattern = re.compile(r"<question>(.*?)</question>", re.DOTALL)
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    qa_list = []

    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all questions and answers in the file
        questions = question_pattern.findall(content)
        answers = answer_pattern.findall(content)

        assert len(questions) == len(answers)

        # Pair questions and answers
        qa_list.extend(zip(map(str.strip, questions), map(str.strip, answers)))
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return qa_list


def parse_questions_answers_with_regex(folder_path):
    """
    Parse question-answer pairs from XML-like text files using regex.

    Args:
        folder_path (str): Path to the folder containing XML-like text files.

    Returns:
        list of tuples: Each tuple contains a question and its corresponding answer.
    """
    # List all text files in the folder
    files = glob(os.path.join(folder_path, "*.txt"))
    qa_list = []

    for file in files:
        qa_list.extend(parse_questions_answers_with_regex_file(file))

    return qa_list


def generate_questions(
    input_folder: str,
    output_folder: str,
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

    questions = parse_questions_answers_with_regex(input_folder)

    if len(questions):
        llm = get_llm_model_chat(temperature=0.1, max_tokens=1000)

        for qa, answ in tqdm(questions):

            # Generate the text using the LLM
            text = llm.invoke(
                [
                    ("system", IMPROVE_QA),
                    ("user", IMPROVE_QA_CONTENT.format(question=qa, answer=answ)),
                ]
            )
            result = {"question": text.content.strip(), "answer": answ}

            # Generate a unique filename for the output
            output_filename = f"{uuid.uuid4().hex}.json"
            output_path = os.path.join(output_folder, output_filename)

            # Save the generated content to the output file
            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(result, out_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions from text files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing input text files of questions",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where output files will be saved.",
    )

    args = parser.parse_args()

    generate_questions(args.input_folder, args.output_folder)
