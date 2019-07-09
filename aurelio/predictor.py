from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from sys import argv
import os

PREDICTOR_NAME = 'bidaf'


def formatInput(passage, question):
    data = {"passage": passage, "question": question}


def predict(model_path, passage, question):
    inputs = {
        "passage": passage,
        "question": question
    }
    model = load_archive(model_path)
    predictor = Predictor.from_archive(model, 'machine-comprehension')
    result = predictor.predict_json(inputs)
    return result
