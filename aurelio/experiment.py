# !!!!IMPORTANT!!!!
# Since we're using the default BiDAF implementation, it's necessary to change spaCy
# word splitter's default language in allennlp's source code at
# allennlp/data/tokenizers/word_splitter.py from "en_core_web_sm" to "pt_core_news_sm"

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


def run(train_dataset_path, dev_dataset_path, portion):
    # Loads files
    with open(train_dataset_path, encoding="utf-8") as file:
        train_dataset = json.loads(file.read())

    with open(dev_dataset_path, encoding="utf-8") as file:
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

    metrics = []

    # Trains a model for each fold with a temporary generated file
    # Training in AllenNLP requires a file
    for train_indexes, dev_indexes in kfold_indexes:

        # config file must be loaded for each iteration
        with open(os.path.realpath("experiment.json"), "r") as file:
            config = json.load(file)

        # GloVe location
        config["model"]["text_field_embedder"]["token_embedders"]["tokens"]["pretrained_file"] \
            = os.path.realpath("glove/glove_s600.zip")

        train = data.iloc[train_indexes, :]
        dev = data.iloc[dev_indexes, :]

        split = list(GroupShuffleSplit(n_splits=1, test_size=portion).split(train, None, train[["context"]]))

        dev = pd.concat([dev, train.iloc[split[0][1], :]])
        train = train.iloc[split[0][0], :]

        # Melts the dataframes into nested datasets
        train_dataset = melt_dataframe(train)
        dev_dataset = melt_dataframe(dev)

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
        model = train_model(params=params,
                            serialization_dir="{}/{}_fold_{}".format(os.path.realpath("serialization"), portion, i))

        # Closes and deletes temp files
        temp_train_file.close()
        temp_dev_file.close()

        metrics.append(model.get_metrics())

        i += 1

    training_f1_scores = []
    dev_f1_scores = []

    training_em_scores = []
    dev_em_scores = []

    for metric in metrics:
        training_f1_scores.append(metric["training_f1"])
        dev_f1_scores.append(metric["validation_f1"])
        training_em_scores.append(metric["training_em"])
        dev_em_scores.append(metric["validation_em"])

    scores = {
        "f1": {
            "train_f1_mean": mean(training_f1_scores),
            "dev_f1_mean": mean(dev_f1_scores),
            "train_f1_stdev": stdev(training_f1_scores),
            "dev_f1_stdev": stdev(dev_f1_scores)
        },
        "em": {
            "train_em_mean": mean(training_em_scores),
            "dev_em_mean": mean(dev_em_scores),
            "train_em_stdev": stdev(training_em_scores),
            "dev_em_stdev": stdev(dev_em_scores)
        }
    }

    with open(os.path.realpath("metrics/portion_{}_metrics.json".format(portion)), "w") as file:
        file.write(json.dumps(scores))

    return scores
