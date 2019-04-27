from allennlp.commands.train import train_model
from allennlp.common.params import Params

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit

from dataset_utils import flatten_json, melt_dataframe, reduce_answer

import pandas as pd
from statistics import mean, stdev
import tempfile
import json
import os


def run(train_dataset_path, dev_dataset_path):
    # Loads files
    with open(train_dataset_path, encoding="utf-8") as file:
        train_dataset = json.loads(file.read())

    with open(dev_dataset_path, encoding="utf-8") as file:
        dev_dataset = json.loads(file.read())

    # dev_dataset = reduce_answer(dev_dataset)

    with open(os.path.realpath("experiment_elmo.json"), "r") as file:
        config = json.load(file)

    # GloVe location
    config["model"]["text_field_embedder"]["token_embedders"]["tokens"]["pretrained_file"] \
        = os.path.realpath("glove/glove_s600.zip")

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
    train_model(params=params,
                serialization_dir="{}/regular".format(os.path.realpath("serialization")))

    # Closes and deletes temp files
    temp_train_file.close()
    temp_dev_file.close()

    print(params)
