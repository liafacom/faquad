import re
import json
import pandas as pd
from os.path import realpath


def get_word_count(text: str):
    preprocessed = re.findall(r'\w+', text)
    return len(preprocessed)


def get_dataset_word_count(dataset: dict):
    context_counts = list()
    question_counts = list()
    answer_counts = list()

    for data in dataset["data"]:
        for paragraph in data["paragraphs"]:
            context_counts.append(get_word_count(paragraph["context"]))

            for qas in paragraph["qas"]:
                question_counts.append(get_word_count(qas["question"]))
                ansbuff = list()

                for answer in qas["answers"]:
                    ansbuff.append(get_word_count(answer["text"]))

                answer_counts += [ansbuff]

    return context_counts, question_counts, answer_counts

