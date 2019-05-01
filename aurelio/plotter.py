import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, iplot
import numpy as np
import json
from statistics import mean, stdev
import os
import collections

GREEN = "rgb(255,122,101)"
GREEN_T = "rgba(255,122,101,0.2)"

RED = "rgb(163,218,151)"
RED_T = "rgba(163,218,151,0.2)"


def plot_single_fold_results(y_train, y_test, y_em_train, y_em_test):
    x_train, x_test = ["10", "25", "50", "75", "100"], ["10", "25", "50", "75", "100"]

    train_f1_plot = go.Scatter(
        x=x_train,
        y=y_train,
        name="Training F1",
        mode='lines',
        line=dict(color=RED)
    )

    dev_f1_plot = go.Scatter(
        x=x_test,
        y=y_test,
        name="Validation F1",
        mode='lines',
        line=dict(color=GREEN)
    )

    train_em_plot = go.Scatter(
        x=x_train,
        y=y_em_train,
        name="Training EM",
        mode='lines',
        line=dict(color=RED, dash="dash")
    )

    dev_em_plot = go.Scatter(
        x=x_test,
        y=y_em_test,
        name="Validation EM",
        mode='lines',
        line=dict(color=GREEN, dash="dash")
    )

    layout = go.Layout(
        title=""
    )

    layout['xaxis'] = dict(title="Training Samples Amount (in %)")
    layout['yaxis'] = dict(title="Score")

    figure = go.Figure(data=[train_f1_plot, dev_f1_plot, train_em_plot, dev_em_plot], layout=layout)
    return py.iplot(figure, filename="c" + ".html")


def plot_results(y_train, y_test, std_train, std_test, y_em_train, y_em_test, std_em_train, std_em_test):
    x_train, x_test = ["10", "25", "50", "75", "100"], ["10", "25", "50", "75", "100"]

    train_f1_plot = go.Scatter(
        x=x_train,
        y=y_train,
        name="Training F1",
        mode='lines',
        line=dict(color=RED)
    )

    dev_f1_plot = go.Scatter(
        x=x_test,
        y=y_test,
        name="Validation F1",
        mode='lines',
        line=dict(color=GREEN)
    )

    train_f1_up_plot = go.Scatter(
        x=x_train,
        y=np.array(y_train) + np.array(std_train),
        showlegend=False,
        fillcolor=RED_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    train_f1_down_plot = go.Scatter(
        x=x_train,
        y=np.array(y_train) - np.array(std_train),
        fill='tonexty',
        name="Training SD",
        fillcolor=RED_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    dev_f1_up_plot = go.Scatter(
        x=x_test,
        y=np.array(y_test) + np.array(std_test),
        name="Validation SD",
        fill='tonexty',
        fillcolor=GREEN_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    dev_f1_down_plot = go.Scatter(
        x=x_test,
        y=np.array(y_test) - np.array(std_test),
        showlegend=False,
        fillcolor=GREEN_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    train_em_plot = go.Scatter(
        x=x_train,
        y=y_em_train,
        name="Training EM",
        mode='lines',
        line=dict(color=RED, dash="dash")
    )

    dev_em_plot = go.Scatter(
        x=x_test,
        y=y_em_test,
        name="Validation EM",
        mode='lines',
        line=dict(color=GREEN, dash="dash")
    )

    train_em_up_plot = go.Scatter(
        x=x_train,
        y=np.array(y_em_train) + np.array(std_em_train),
        showlegend=False,
        fillcolor=RED_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    train_em_down_plot = go.Scatter(
        x=x_train,
        y=np.array(y_em_train) - np.array(std_em_train),
        fill='tonexty',
        name="Training EM SD",
        showlegend=False,
        fillcolor=RED_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    dev_em_up_plot = go.Scatter(
        x=x_test,
        y=np.array(y_em_test) + np.array(std_em_test),
        name="Validation EM SD",
        showlegend=False,
        fill='tonexty',
        fillcolor=GREEN_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    dev_em_down_plot = go.Scatter(
        x=x_test,
        y=np.array(y_em_test) - np.array(std_em_test),
        showlegend=False,
        fillcolor=GREEN_T,
        line=dict(color='rgba(255,255,255,0)')
    )

    layout = go.Layout(
        title=""
    )
    layout['xaxis'] = dict(title="Training Samples Amount (in %)")
    layout['yaxis'] = dict(title="Score")

    figure = go.Figure(data=[train_f1_plot, dev_f1_plot, train_f1_up_plot,
                             train_f1_down_plot, dev_f1_down_plot,
                             dev_f1_up_plot, train_em_plot, dev_em_plot, train_em_up_plot,
                             train_em_down_plot, dev_em_down_plot,
                             dev_em_up_plot], layout=layout)
    return py.iplot(figure, filename="c" + ".html")


def plot_single_fold(metrics_dir):
    scores = collections.OrderedDict({
        "0.9": {
            "f1": {
                "train": 0.0,
                "dev": 0.0
            },
            "em": {
                "train": 0.0,
                "dev": 0.0
            }
        },
        "0.75": {
            "f1": {
                "train": 0.0,
                "dev": 0.0
            },
            "em": {
                "train": 0.0,
                "dev": 0.0
            }
        },
        "0.5": {
            "f1": {
                "train": 0.0,
                "dev": 0.0
            },
            "em": {
                "train": 0.0,
                "dev": 0.0
            }
        },
        "0.25": {
            "f1": {
                "train": 0.0,
                "dev": 0.0
            },
            "em": {
                "train": 0.0,
                "dev": 0.0
            }
        },
        "0.0": {
            "f1": {
                "train": 0.0,
                "dev": 0.0
            },
            "em": {
                "train": 0.0,
                "dev": 0.0
            }
        }
    })

    for filename in os.listdir(metrics_dir):
        with open("{}/{}/metrics.json".format(metrics_dir, filename)) as file:
            metrics = json.loads(file.read())

            if "0.0" in filename:
                scores["0.0"]["f1"]["train"] = metrics["training_f1"]
                scores["0.0"]["f1"]["dev"] = metrics["best_validation_f1"]
                scores["0.0"]["em"]["train"] = metrics["training_em"]
                scores["0.0"]["em"]["dev"] = metrics["best_validation_em"]

            if "0.25" in filename:
                scores["0.25"]["f1"]["train"] = metrics["training_f1"]
                scores["0.25"]["f1"]["dev"] = metrics["best_validation_f1"]
                scores["0.25"]["em"]["train"] = metrics["training_em"]
                scores["0.25"]["em"]["dev"] = metrics["best_validation_em"]

            if "0.5" in filename:
                scores["0.5"]["f1"]["train"] = metrics["training_f1"]
                scores["0.5"]["f1"]["dev"] = metrics["best_validation_f1"]
                scores["0.5"]["em"]["train"] = metrics["training_em"]
                scores["0.5"]["em"]["dev"] = metrics["best_validation_em"]

            if "0.75" in filename:
                scores["0.75"]["f1"]["train"] = metrics["training_f1"]
                scores["0.75"]["f1"]["dev"] = metrics["best_validation_f1"]
                scores["0.75"]["em"]["train"] = metrics["training_em"]
                scores["0.75"]["em"]["dev"] = metrics["best_validation_em"]

            if "0.9" in filename:
                scores["0.9"]["f1"]["train"] = metrics["training_f1"]
                scores["0.9"]["f1"]["dev"] = metrics["best_validation_f1"]
                scores["0.9"]["em"]["train"] = metrics["training_em"]
                scores["0.9"]["em"]["dev"] = metrics["best_validation_em"]

    scores_train_f1 = []
    scores_train_em = []
    scores_dev_f1 = []
    scores_dev_em = []

    for key in scores:
        scores_train_f1.append(scores[key]["f1"]["train"])
        scores_train_em.append(scores[key]["em"]["train"])
        scores_dev_f1.append(scores[key]["f1"]["dev"])
        scores_dev_em.append(scores[key]["em"]["dev"])

    return plot_single_fold_results(scores_train_f1, scores_dev_f1, scores_train_em, scores_dev_em)


def plot(metrics_dir):
    scores = {
        "0.0": {
            "f1": {
                "train": [],
                "dev": []
            },
            "em": {
                "train": [],
                "dev": []
            }
        },
        "0.25": {
            "f1": {
                "train": [],
                "dev": []
            },
            "em": {
                "train": [],
                "dev": []
            }
        },
        "0.5": {
            "f1": {
                "train": [],
                "dev": []
            },
            "em": {
                "train": [],
                "dev": []
            }
        },
        "0.75": {
            "f1": {
                "train": [],
                "dev": []
            },
            "em": {
                "train": [],
                "dev": []
            }
        },
        "0.9": {
            "f1": {
                "train": [],
                "dev": []
            },
            "em": {
                "train": [],
                "dev": []
            }
        }
    }

    for filename in os.listdir(metrics_dir):
        with open("{}/{}/metrics.json".format(metrics_dir, filename)) as file:
            metrics = json.loads(file.read())

            if "0.0" in filename:
                scores["0.0"]["f1"]["train"].append(metrics["training_f1"])
                scores["0.0"]["f1"]["dev"].append(metrics["validation_f1"])
                scores["0.0"]["em"]["train"].append(metrics["training_em"])
                scores["0.0"]["em"]["dev"].append(metrics["validation_em"])

            if "0.25" in filename:
                scores["0.25"]["f1"]["train"].append(metrics["training_f1"])
                scores["0.25"]["f1"]["dev"].append(metrics["validation_f1"])
                scores["0.25"]["em"]["train"].append(metrics["training_em"])
                scores["0.25"]["em"]["dev"].append(metrics["validation_em"])

            if "0.5" in filename:
                scores["0.5"]["f1"]["train"].append(metrics["training_f1"])
                scores["0.5"]["f1"]["dev"].append(metrics["validation_f1"])
                scores["0.5"]["em"]["train"].append(metrics["training_em"])
                scores["0.5"]["em"]["dev"].append(metrics["validation_em"])

            if "0.75" in filename:
                scores["0.75"]["f1"]["train"].append(metrics["training_f1"])
                scores["0.75"]["f1"]["dev"].append(metrics["validation_f1"])
                scores["0.75"]["em"]["train"].append(metrics["training_em"])
                scores["0.75"]["em"]["dev"].append(metrics["validation_em"])

            if "0.9" in filename:
                scores["0.9"]["f1"]["train"].append(metrics["training_f1"])
                scores["0.9"]["f1"]["dev"].append(metrics["validation_f1"])
                scores["0.9"]["em"]["train"].append(metrics["training_em"])
                scores["0.9"]["em"]["dev"].append(metrics["validation_em"])

    train_f1_means = [
        mean(scores["0.9"]["f1"]["train"]),
        mean(scores["0.75"]["f1"]["train"]),
        mean(scores["0.5"]["f1"]["train"]),
        mean(scores["0.25"]["f1"]["train"]),
        mean(scores["0.0"]["f1"]["train"])
    ]
    train_f1_stdev = [
        stdev(scores["0.9"]["f1"]["train"]),
        stdev(scores["0.75"]["f1"]["train"]),
        stdev(scores["0.5"]["f1"]["train"]),
        stdev(scores["0.25"]["f1"]["train"]),
        stdev(scores["0.0"]["f1"]["train"])
    ]
    dev_f1_means = [
        mean(scores["0.9"]["f1"]["dev"]),
        mean(scores["0.75"]["f1"]["dev"]),
        mean(scores["0.5"]["f1"]["dev"]),
        mean(scores["0.25"]["f1"]["dev"]),
        mean(scores["0.0"]["f1"]["dev"])
    ]
    dev_f1_stdev = [
        stdev(scores["0.9"]["f1"]["dev"]),
        stdev(scores["0.75"]["f1"]["dev"]),
        stdev(scores["0.5"]["f1"]["dev"]),
        stdev(scores["0.25"]["f1"]["dev"]),
        stdev(scores["0.0"]["f1"]["dev"])
    ]
    train_em_means = [
        mean(scores["0.9"]["em"]["train"]),
        mean(scores["0.75"]["em"]["train"]),
        mean(scores["0.5"]["em"]["train"]),
        mean(scores["0.25"]["em"]["train"]),
        mean(scores["0.0"]["em"]["train"])
    ]
    train_em_stdev = [
        stdev(scores["0.9"]["em"]["train"]),
        stdev(scores["0.75"]["em"]["train"]),
        stdev(scores["0.5"]["em"]["train"]),
        stdev(scores["0.25"]["em"]["train"]),
        stdev(scores["0.0"]["em"]["train"])
    ]
    dev_em_means = [
        mean(scores["0.9"]["em"]["dev"]),
        mean(scores["0.75"]["em"]["dev"]),
        mean(scores["0.5"]["em"]["dev"]),
        mean(scores["0.25"]["em"]["dev"]),
        mean(scores["0.0"]["em"]["dev"])
    ]
    dev_em_stdev = [
        stdev(scores["0.9"]["em"]["dev"]),
        stdev(scores["0.75"]["em"]["dev"]),
        stdev(scores["0.5"]["em"]["dev"]),
        stdev(scores["0.25"]["em"]["dev"]),
        stdev(scores["0.0"]["em"]["dev"])
    ]

    return plot_results(train_f1_means, dev_f1_means, train_f1_stdev, dev_f1_stdev,
                        train_em_means, dev_em_means, train_em_stdev, dev_em_stdev)
