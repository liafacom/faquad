# !!!!IMPORTANT!!!!
# Since we're using the default BiDAF implementation, it's necessary to change spaCy
# word splitter's default language in allennlp's source code at
# allennlp/data/tokenizers/word_splitter.py from "en_core_web_sm" to "pt_core_news_sm"

from allennlp.commands import main

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from dataset_utils import flatten_json, melt_dataframe, reduce_answer

import pandas as pd
import tempfile
import json
import shutil
import os
import sys


def run_train(config_file_path, train_dataset_path, dev_dataset_path, serialization_dir, elmo=True):
    if elmo:
        overrides = {
            "train_data_path": train_dataset_path,
            "validation_data_path": dev_dataset_path,
            "dataset_reader": {
                "token_indexers": {
                    "elmo": {
                        "type": "elmo_characters"
                    }
                }
            },
            "model": {
                "text_field_embedder": {
                    "token_embedders": {
                        "elmo": {
                            "type": "elmo_token_embedder",
                            "options_file": "elmo/elmo_pt_options.json",
                            "weight_file": "elmo/elmo_pt_weights.hdf5",
                            "do_layer_norm": False,
                            "dropout": 0.0
                        }
                    }
                },
                "phrase_layer": {
                    "input_size": 1724
                }
            }
        }
    else:
        overrides = {
            "train_data_path": train_dataset_path,
            "validation_data_path": dev_dataset_path
        }

    shutil.rmtree(serialization_dir, ignore_errors=True)

    sys.argv = [
        "allennlp",
        "train",
        config_file_path,
        "-s", serialization_dir,
        "-o", json.dumps(overrides)
    ]

    main()


def run_single_fold(config_file_path,
                    train_dataset_path,
                    dev_dataset_path,
                    serialization_dir,
                    reduce_train_dataset=True,
                    elmo=True,
                    dev_dataset_portion=0.0):
    # Loads files
    with open(train_dataset_path, encoding="utf-8") as file:
        train_dataset = json.loads(file.read())

    with open(dev_dataset_path, encoding="utf-8") as file:
        dev_dataset = json.loads(file.read())

    # Flattens nested datasets into dataframes
    train = flatten_json(train_dataset)
    dev = flatten_json(dev_dataset)

    split = list(
        ShuffleSplit(n_splits=1, test_size=dev_dataset_portion).split(train))

    train = train.iloc[split[0][0], :]

    # Melts the dataframes into nested datasets
    train_dataset = reduce_answer(melt_dataframe(train)) if reduce_train_dataset else melt_dataframe(train)
    dev_dataset = melt_dataframe(dev)

    # Writes a temporary training file
    temp_train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="UTF-8")
    temp_train_file.write(json.dumps(train_dataset))
    print("Temp file wrote at {}".format(temp_train_file.name))

    # Writes a temporary dev file
    temp_dev_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="UTF-8")
    temp_dev_file.write(json.dumps(dev_dataset))
    print("Temp file wrote at {}".format(temp_dev_file.name))

    run_train(config_file_path,
              temp_train_file.name,
              temp_dev_file.name,
              "{}/{}_{}_fold_{}".format(os.path.realpath(serialization_dir),
                                        "elmo" if elmo else "no_elmo",
                                        dev_dataset_portion,
                                        0),
              elmo)

    # Closes and deletes temp files
    temp_train_file.close()
    temp_dev_file.close()


def run_kfold(config_file_path,
              train_dataset_path,
              dev_dataset_path,
              serialization_dir,
              reduce_train_dataset=True,
              elmo=True,
              dev_dataset_portion=0.0):
    # Loads files
    with open(train_dataset_path, encoding="utf-8") as file:
        train_dataset = json.loads(file.read())

    with open(dev_dataset_path, encoding="utf-8") as file:
        dev_dataset = json.loads(file.read())

    # Flattens nested datasets into dataframes
    train_data = flatten_json(train_dataset)
    dev_data = flatten_json(dev_dataset)

    # Concatenates train and dev dataframes
    data = pd.concat([train_data, dev_data])

    # Configures GroupKFold and select indexes
    kfold = GroupKFold(n_splits=10)
    kfold_indexes = kfold.split(data, None, data[["context"]])

    i = 1

    # Trains a model for each fold with a temporary generated file
    # Training in AllenNLP requires a file
    for train_indexes, dev_indexes in kfold_indexes:
        train = data.iloc[train_indexes, :]
        dev = data.iloc[dev_indexes, :]

        split = list(
            GroupShuffleSplit(n_splits=1, test_size=dev_dataset_portion).split(train, None, train[["context"]]))

        train = train.iloc[split[0][0], :]

        # Melts the dataframes into nested datasets
        train_dataset = reduce_answer(melt_dataframe(train)) if reduce_train_dataset else melt_dataframe(train)
        dev_dataset = melt_dataframe(dev)

        # Writes a temporary training file
        temp_train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="UTF-8")
        temp_train_file.write(json.dumps(train_dataset))
        print("Temp file wrote at {}".format(temp_train_file.name))

        # Writes a temporary dev file
        temp_dev_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="UTF-8")
        temp_dev_file.write(json.dumps(dev_dataset))
        print("Temp file wrote at {}".format(temp_dev_file.name))

        run_train(config_file_path,
                  temp_train_file.name,
                  temp_dev_file.name,
                  "{}/{}_{}_fold_{}".format(os.path.realpath(serialization_dir),
                                            "elmo" if elmo else "no_elmo",
                                            dev_dataset_portion,
                                            i),
                  elmo)

        # Closes and deletes temp files
        temp_train_file.close()
        temp_dev_file.close()

        i += 1
