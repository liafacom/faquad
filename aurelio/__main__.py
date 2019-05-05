from experiment import run_kfold, run_single_fold

import os

GREEN = "rgb(163,218,151)"
GREEN_T = "rgba(163,218,151,0.2)"

RED = "rgb(255,122,101)"
RED_T = "rgba(255,122,101,0.2)"


def main():
    # Portions relative to TEST size

    embedding_dims = [50, 100, 300, 600]

    for dim in embedding_dims:
        run_kfold(os.path.realpath("experiment.json"),
                  os.path.realpath("data/qa_facom_dataset_train_increased.json"),
                  os.path.realpath("data/qa_facom_dataset_dev.json"),
                  os.path.realpath("experiment_glove_dimensions"),
                  True,
                  False,
                  0.0,
                  dim)


main()
