from experiment import run
import plotly.plotly as py
import plotly.graph_objs as go

import os


def main():
    # Portions relative to TEST size
    portions = [0.0, 0.1, 0.25, 0.5, 0.75]

    for portion in portions:
        run(os.path.realpath("data/qa_facom_dataset_train.json".format(portion)),
            os.path.realpath("data/qa_facom_dataset_dev.json".format(portion)),
            portion)


main()
