import json

with open("increased_ds/qa_facom_dataset_train_increased.json") as file:
    ds = json.loads(file.read())

with open("data/qa_facom_dataset_train.json") as file:
    ods = json.loads(file.read())


def insertAnswers(node, answers):
    if "id" in node and node["id"] == answers["id"]:
        node["answers"] = answers["answers"]

    for key in node:
        if isinstance(node[key], list):
            for obj in node[key]:
                insertAnswers(obj, answers)


for obj in ds:
    insertAnswers(ods, obj);

with open("data/qa_facom_dataset_train_increased.json", "w") as file:
    file.write(json.dumps(ods))