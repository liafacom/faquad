from pandas.io.json import json_normalize


def expand_questions(node: list):
    for data in node:
        while len(data["answers"]) > 1:
            node.append({
                "question": data["question"],
                "id": data["id"],
                "answers": [data["answers"].pop()]
            })


def expand_qas(node):
    if not isinstance(node, list):
        return

    for data in node:
        keys = data.keys()

        if "qas" in keys:
            expand_questions(data["qas"])
        else:
            for key in keys:
                expand_qas(data[key])


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
                if k == "answer_start":
                    obj[key_alias[index]] = int(name[index])
                else:
                    obj[key_alias[index]] = name[index]
            if payload_key is not None:
                obj[payload_key] = group

        result.append(obj)
    return result


def reduce_answer(dataset):
    dataset["data"] = reduce(dataset["data"], ["paragraphs", "qas", "answers"])
    return dataset


def reduce(array, key_sequence):
    if len(key_sequence) == 0:
        return [array.pop(0)]

    for obj in array:
        key = key_sequence[0]
        obj[key] = reduce(obj[key], key_sequence[1:])

    return array
