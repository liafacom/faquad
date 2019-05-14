from experiment import run_kfold
from os.path import realpath
from sys import argv


def main():
    # Portions relative to Group Shuffle Split TEST size
    portions = [0.0, 0.25, 0.5, 0.75, 0.9]

    for portion in portions:
        run_kfold(config_file_path=realpath("experiment.json"),
                  train_dataset_path=realpath("data/train.json"),
                  dev_dataset_path=realpath("data/dev.json"),
                  serialization_dir=realpath("models/{}".format(argv[1])),
                  reduce_train_dataset=True,
                  reduce_dev_dataset=False,
                  expand_train_qas=False,
                  elmo=True,
                  dev_dataset_portion=portion,
                  embedding_dim=100)


main()
