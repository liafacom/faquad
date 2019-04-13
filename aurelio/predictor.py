from allennlp.service.predictors import Predictor
from allennlp.models.archival import load_archive

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
    result = predictor.predict_json(inputs);
    return result


result = predict("/home/helio/Documentos/chatbot_dataset/experiments/scripts/bse_50_2.tar.gz", questions[0]['context'],
                 questions[0]['question'])
print(result)
