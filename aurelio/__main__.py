from experiment import run
import os


def main():
    portions = ["10", "25", "50", "75", "100"]

    for portion in portions:
        run(os.path.realpath("data/train/qa_facom_dataset_{}.json".format(portion)),
            os.path.realpath("data/dev/qa_facom_dataset_{}.json".format(portion)),
            portion)

main()