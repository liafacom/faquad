from allennlp.service.predictors import Predictor
from allennlp.models.archival import load_archive
from contextlib import contextmanager
from sys import argv
import os
import sys

PREDICTOR_NAME = 'bidaf'

def formatInput(passage, question):
    data={"passage":passage, "question":question}

def predict(model_path, passage, question):
    inputs = {
        "passage":passage,
        "question":question
    }
    model = load_archive(model_path)
    predictor = Predictor.from_archive(model, 'machine-comprehension')
    result = predictor.predict_json(inputs);
    return result

# result = predict(argv[1], argv[2], argv[3])
# print(result['best_span_str']);
questions = [
{
    "context":"o acadêmico terá direito à revisão de suas avaliações acadêmicas dirigindo-se ao professor, em primeira instância, por meio de requerimento protocolizado na secretaria acadêmica da unidade da administração setorial em que o curso é oferecido, no prazo de três dias úteis após a divulgação do resultado. o professor terá o prazo de dois dias úteis para manifestação escrita sobre o pedido. o acadêmico deverá apor seu ciente no documento de resposta, e receber uma cópia deste. o acadêmico poderá interpor recurso quanto ao resultado da revisão, no colegiado de curso, via secretaria acadêmica, no prazo de cinco dias úteis do seu ciente. o colegiado de curso deverá constituir comissão, composta por três docentes, preferencialmente da área, sendo vedada a inclusão do professor que corrigiu a avaliação acadêmica em questão. a comissão deverá analisar o pedido do acadêmico, consultar o professor, se necessário, e emitir parecer sobre o resultado da revisão, num prazo máximo de quinze dias a partir da publicação da resolução de constituição da comissão, e encaminhar para aprovação do colegiado de curso. o professor da disciplina será responsável pela alteração no siscad, em caso de modificação da nota resultante dos trabalhos da comissão de revisão.",
    "question":"Por quantos docentes deve ser composta a comissão?"
}]

result = predict("/home/helio/Documentos/chatbot_dataset/experiments/scripts/bse_50_2.tar.gz", questions[0]['context'], questions[0]['question'])
# print(questions[0])
print(result)

