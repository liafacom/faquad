from pandas.io.json import json_normalize


def flatten_json(dict):
    dataframe = json_normalize(dict["data"], ["paragraphs", 'qas', "answers"],
                               ["title", ["paragraphs", "context"], ["paragraphs", "qas", "question"],
                                ["paragraphs", "qas", "id"]]).rename(
        columns={"text": "answer", "paragraphs.context": "context", "paragraphs.qas.question": "question",
                 "paragraphs.qas.id": "id"})
    dataframe["answer_start"] = dataframe["answer_start"].astype('str')
    return dataframe


def melt_dataframe(dataframe):
    dataset = {"data": melt(dataframe, "title", "title", "paragraphs"), "version": 1.0}

    for title in dataset["data"]:
        title["paragraphs"] = melt(title["paragraphs"], "context", "context", "qas")
        for paragraph in title["paragraphs"]:
            paragraph["qas"] = melt(paragraph["qas"], ["question", "id"], ["question", "id"], "answers")
            for question in paragraph["qas"]:
                question["answers"] = melt(question["answers"], ["answer", "answer_start"], ["text", "answer_start"],
                                           None)

    return dataset


def melt(dataframe, key, key_alias, payload_key):
    result = []
    groups = dataframe.groupby(key)
    for name, group in groups:
        obj = {}

        if isinstance(key, str):
            group.pop(key)
            obj[key_alias] = name
            if payload_key is not None:
                obj[payload_key] = group
        elif isinstance(key, list):
            for index, k in enumerate(key):
                group.pop(k)
                obj[key_alias[index]] = name[index]
            if payload_key is not None:
                obj[payload_key] = group

        result.append(obj)
    return result
