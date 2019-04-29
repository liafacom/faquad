from experiment import run_kfold

import os

GREEN = "rgb(163,218,151)"
GREEN_T = "rgba(163,218,151,0.2)"

RED = "rgb(255,122,101)"
RED_T = "rgba(255,122,101,0.2)"


def main():
    # Portions relative to TEST size
    portions = [0.0, 0.25, 0.5, 0.75, 0.9]

    for portion in portions:
        run_kfold(os.path.realpath("experiment.json"),
            os.path.realpath("data/qa_facom_dataset_train_increased.json"),
            os.path.realpath("data/qa_facom_dataset_dev.json"),
            os.path.realpath("experiments"),
            False,
            True,
            portion)


main()
