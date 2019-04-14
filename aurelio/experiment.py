# !!!!IMPORTANT!!!!
# Since we're using the default BiDAF implementation, it's necessary to change spaCy
# word splitter's default language in allennlp's source code at
# allennlp/data/tokenizers/word_splitter.py from "en_core_web_sm" to "pt_core_news_sm"

from allennlp.commands.train import train_model
from allennlp.common.params import Params

from sklearn.model_selection import GroupKFold

from dataset_utils import flatten_json, melt_dataframe, reduce_answer

import pandas as pd

import tempfile
import json
import os

# Loads files
with open(os.path.realpath("data/qa_facom_dataset_837.json"), encoding="utf-8") as file:
    train_dataset = json.loads(file.read())

with open(os.path.realpath("data/qa_facom_dev.json"), encoding="utf-8") as file:
    dev_dataset = json.loads(file.read())

# Flattens nested datasets into dataframes
train_data = flatten_json(train_dataset)
dev_data = flatten_json(reduce_answer(dev_dataset))

# Concatenates train and dev dataframes
data = pd.concat([train_data, dev_data])

# Configures GroupKFold and select indexes
kfold = GroupKFold(n_splits=10)
kfold_indexes = kfold.split(data, None, data[["context"]])

i = 1

# Trains a model for each fold with a temporary generated file
# Training in AllenNLP requires a file
for train_indexes, dev_indexes in kfold_indexes:

    with open(os.path.realpath("experiment.json"), "r") as file:
        config = json.load(file)

    # GloVe location
    config["model"]["text_field_embedder"]["token_embedders"]["tokens"]["pretrained_file"] \
        = os.path.realpath("glove/glove_s600.zip")

    # Melts the dataframes into nested datasets
    train_dataset = melt_dataframe(data.iloc[train_indexes, :])
    dev_dataset = melt_dataframe(data.iloc[dev_indexes, :])

    # Writes a temporary training file
    temp_train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="UTF-8")
    temp_train_file.write(json.dumps(train_dataset))
    print("Temp file wrote at {}".format(temp_train_file.name))

    # Writes a temporary dev file
    temp_dev_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="UTF-8")
    temp_dev_file.write(json.dumps(dev_dataset))
    print("Temp file wrote at {}".format(temp_dev_file.name))

    # Temporary dataset files location
    config["train_data_path"] = temp_train_file.name
    config["validation_data_path"] = temp_dev_file.name

    # Creates a Param class and writes the train result
    params = Params(config)
    train_model(params=params, serialization_dir="{}/fold_{}".format(os.path.realpath("serialization"), i))

    # Closes and deletes temp files
    temp_train_file.close()
    temp_dev_file.close()

    i += 1