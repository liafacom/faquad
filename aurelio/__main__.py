from experiment import run
import regular_training
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

import os

GREEN = "rgb(163,218,151)"
GREEN_T = "rgba(163,218,151,0.2)"

RED = "rgb(255,122,101)"
RED_T = "rgba(255,122,101,0.2)"

def main():
    # Portions relative to TEST size
    portions = [0.0, 0.25, 0.5, 0.75, 0.9]

    # scores = []

    for portion in portions:
         run(os.path.realpath("data/qa_facom_dataset_train_increased.json".format(portion)),
             os.path.realpath("data/qa_facom_dataset_dev.json".format(portion)),
             "experiment_elmo.json",
             "experiment_elmo",
             portion)

        # scores.append(score)

    # regular_training.run("data/qa_facom_dataset_train.json", "data/qa_facom_dataset_dev.json")

    # x = ["100%", "75%", "50%", "25%", "10%"]
    #
    # train_f1_means = []
    # train_f1_stdev = []
    #
    # dev_f1_means = []
    # dev_f1_stdev = []
    #
    # train_em_means = []
    # train_em_stdev = []
    #
    # dev_em_means = []
    # dev_em_stdev = []
    #
    # for score in scores:
    #     train_f1_means.append(score["train_f1_mean"])
    #     train_f1_stdev.append(score["train_f1_stdev"])
    #
    #     dev_f1_means.append(score["dev_f1_mean"])
    #     dev_f1_stdev.append(score["dev_f1_stdev"])
    #
    #     train_em_means.append(score["train_em_mean"])
    #     train_em_stdev.append(score["train_em_stdev"])
    #
    #     dev_em_means.append(score["dev_em_mean"])
    #     dev_em_stdev.append(score["dev_em_stdev"])
    #
    # train_f1_stdev_up = np.array(train_f1_means) + np.array(train_f1_stdev)
    # train_f1_stdev_down = np.array(train_f1_means) - np.array(train_f1_stdev)
    #
    # train_em_stdev_up = np.array(train_em_means) + np.array(train_em_stdev)
    # train_em_stdev_down = np.array(train_em_means) - np.array(train_em_stdev)
    #
    # dev_f1_stdev_up = np.array(dev_f1_means) + np.array(dev_f1_stdev)
    # dev_f1_stdev_down = np.array(dev_f1_means) - np.array(dev_f1_stdev)
    #
    # dev_em_stdev_up = np.array(dev_em_means) + np.array(dev_em_stdev)
    # dev_em_stdev_down = np.array(dev_em_means) - np.array(dev_em_stdev)
    #
    # train_f1_stdev_plot = go.Scatter(
    #     x=x + x[::-1],
    #     y=train_f1_stdev_up + train_f1_stdev_down,
    #     fill='tozerox',
    #     fillcolor=RED_T,
    #     line=dict(color='rgba(255,255,255,0)'),
    #     showlegend=False,
    #     name='Train F1 Standard Deviation',
    # )
    #
    # train_em_stdev_plot = go.Scatter(
    #     x=x + x[::-1],
    #     y=train_em_stdev_up + train_em_stdev_down,
    #     fill='tozerox',
    #     fillcolor=RED_T,
    #     line=dict(color='rgba(255,255,255,0)'),
    #     showlegend=False,
    #     name='Train EM Standard Deviation',
    # )
    #
    # dev_f1_stdev_plot = go.Scatter(
    #     x=x + x[::-1],
    #     y=dev_f1_stdev_up + dev_f1_stdev_down,
    #     fill='tozerox',
    #     fillcolor=GREEN_T,
    #     line=dict(color='rgba(255,255,255,0)'),
    #     showlegend=False,
    #     name='Validation F1 Standard Deviation',
    # )
    #
    # dev_em_stdev_plot = go.Scatter(
    #     x=x + x[::-1],
    #     y=dev_em_stdev_up + dev_em_stdev_down,
    #     fill='tozerox',
    #     fillcolor=GREEN_T,
    #     line=dict(color='rgba(255,255,255,0)'),
    #     showlegend=False,
    #     name='Validation EM Standard Deviation',
    # )
    #
    # train_f1_plot = go.Scatter(
    #     x=x,
    #     y=train_f1_means,
    #     line=dict(color=RED),
    #     mode='lines',
    #     name='Train F1 Mean',
    # )
    #
    # train_em_plot = go.Scatter(
    #     x=x,
    #     y=train_em_means,
    #     line=dict(color=RED, dash="dash"),
    #     mode='lines',
    #     name='Train EM Mean',
    # )
    #
    # dev_f1_plot = go.Scatter(
    #     x=x,
    #     y=dev_f1_means,
    #     line=dict(color=GREEN),
    #     mode='lines',
    #     name='Validation F1 Mean',
    # )
    #
    # dev_em_plot = go.Scatter(
    #     x=x,
    #     y=dev_em_means,
    #     line=dict(color=GREEN, dash="dash"),
    #     mode='lines',
    #     name='Validation EM Mean',
    # )
    #
    # data = [train_f1_stdev_plot, train_em_stdev_plot, dev_f1_stdev_plot, dev_em_stdev_plot,
    #         train_f1_plot, train_em_plot, dev_f1_plot, dev_em_plot]
    #
    # layout = go.Layout(
    #     paper_bgcolor='rgb(255,255,255)',
    #     plot_bgcolor='rgb(229,229,229)',
    #     xaxis=dict(
    #         gridcolor='rgb(255,255,255)',
    #         range=[1, 10],
    #         showgrid=True,
    #         showline=False,
    #         showticklabels=True,
    #         tickcolor='rgb(127,127,127)',
    #         ticks='outside',
    #         zeroline=False
    #     ),
    #     yaxis=dict(
    #         gridcolor='rgb(255,255,255)',
    #         showgrid=True,
    #         showline=False,
    #         showticklabels=True,
    #         tickcolor='rgb(127,127,127)',
    #         ticks='outside',
    #         zeroline=False
    #     ),
    # )
    #
    # py.iplot(dict(data=data), filename="kfold_plot")
    #

main()
