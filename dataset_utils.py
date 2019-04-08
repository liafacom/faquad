from pandas.io.json import json_normalize

def flatten_json(dict):
	return json_normalize(dict["data"], ["paragraphs", 'qas', "answers"], ["title", ["paragraphs", "context"], ["paragraphs", "qas", "question"]]).rename(columns={"text": "answer", "paragraphs.context": "context", "paragraphs.qas.question": "question"})

def get_dataframe_folds(dataframe, indexes):
	return None
	
def transform_dataframe_folds(dataframe, indexes):
	return None