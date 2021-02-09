from collections import OrderedDict
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from make_hinton_diagram import NestedDictValues
import pdb
import matplotlib.pyplot as plt

model_order = [
        "1NN",
        "2NN",
        "3NN",
        "4NN",
        "5NN", 
        "Progenitor",
        "Prototype",
        "Exemplar",
        #"Exemplar (s=1)",
        "Local"
    ]

def summarize_rank_diffs(data: dict, p_vals: bool = False) -> tuple:
    data = OrderedDict(data)
    del data["Exemplar (s=1)"] # no longer including in figs
    vals = list(NestedDictValues(data))
    num_models = len(model_order)
    num_scientists = len(data["1NN"])
    scientist_order = sorted(list(data[model_order[0]].keys()))
    #rank_diffs = np.array([val[0] for val in vals]).reshape(num_models, num_scientists)

    rank_diffs = np.zeros((num_models, num_scientists))
    for i, model in enumerate(model_order):
        for j, scientist in enumerate(scientist_order):
            rank_diffs[i][j] = data[model][scientist][0]

    if p_vals:
        p_vals = np.array([val[1] * num_scientists for val in vals]).reshape(num_models, num_scientists)
    else:
        p_vals = np.ones((num_models, num_scientists))
    models = model_order
    names = [name.replace("=", " ").replace("_", " ").replace("-", " ").replace("0001", "") for name in scientist_order]
    names = [name if "Fernando" not in name else "Fernando J Corbató" for name in names]
    print(rank_diffs)
    #maxscale = np.amax(rank_diffs, axis=1).reshape(-1, 1)
    maxscale = np.tile(np.amax(rank_diffs, axis=0).reshape(1, -1), (rank_diffs.shape[0], 1))
    #minscale = np.amin(rank_diffs, axis=1).reshape(-1, 1)
    minscale = np.tile(np.amin(rank_diffs, axis=0).reshape(1, -1), (rank_diffs.shape[0], 1))
    print(maxscale)
    #pdb.set_trace()
    #pdb.set_trace()
    print(rank_diffs.shape)
    print(maxscale.shape)

    rank_diffs = np.divide(rank_diffs - minscale, maxscale - minscale)
    np.nan_to_num(rank_diffs)
    #rank_diffs = rank_diffs / rank_diffs.max(axis=0)
    rank_diffs = pd.DataFrame(rank_diffs.T, columns=model_order)
    print(rank_diffs)
    #rank_diffs = rank_diffs.div(rank_diffs.sum(axis=1), axis=0)

    print(np.max(rank_diffs, axis=1))

    return p_vals, rank_diffs, names

def violin_plot(rank_diffs: np.ndarray) -> None:
    rank_diffs_line = rank_diffs.copy()
    rank_diffs_line["id"] = np.arange(0, rank_diffs.shape[0])
    rank_diffs_line = rank_diffs_line.melt(var_name="model", value_name="vals", id_vars="id")
    rank_diffs = rank_diffs.melt(var_name="model", value_name="vals")
    ax = sns.violinplot(x="model", y="vals", data=rank_diffs, cut=0, inner="point")
    plt.xlabel("Model")
    plt.ylabel("Performance relative to best model (1.0)\nand worst model (0.0)")
    #sns.lineplot(x="model", y="vals", data=rank_diffs_line, markers=True, units="id", estimator=None, hue="id", palette=sns.color_palette("RdBu", n_colors=rank_diffs_line["id"].max() + 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig("testing-violin-plot-econ-2.png")

if __name__ == "__main__":
    models = pickle.load(open("results/summary/cv-final/economics-2.p", "rb"))
    _, rank_diffs, name = summarize_rank_diffs(models, p_vals=False)
    violin_plot(rank_diffs)