from collections import OrderedDict
from typing import List
import pickle
import pdb

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src.utils.hinton import hinton

model_order = [
        "1NN",
        #"2NN",
        #"3NN",
        #"4NN",
        #"5NN", 
        "Progenitor",
        "Prototype",
        "Exemplar",
        #"Exemplar (s=1)",
        "Local"
    ]

def find_rank_diff(filenames: List) -> dict:
    raise NotImplementedError

# data format: {model: {name: (rank diff, p value)}}
def make_heatmap(data: dict, filename: str, p_vals=True, labelled=False, measure_type="rank", include_scientist_names=True) -> None:
    del data["Null"]
    data = OrderedDict(data)
    if labelled: # order is p_val, win_ratio, yerr, tval, num_papers
        vals = list(NestedDictValues(data))[::5]
    else:    
        vals = list(NestedDictValues(data))

    num_models = len(data)
    num_scientists = len(data["1NN"])
    print(num_scientists)
    scientist_order = sorted(list(data[model_order[0]].keys()))
    #rank_diffs = np.array([val[0] for val in vals]).reshape(num_models, num_scientists)
    num_above_chance = 0
    seen = set()

    rank_diffs = np.zeros((num_models, num_scientists))
    for i, model in enumerate(model_order):
        for j, scientist in enumerate(scientist_order):
            if labelled:
                if measure_type == "rank":
                    rank_diffs[i][j] = data[model][scientist]["win_ratio"] - 0.5
                    if rank_diffs[i][j] > 0 and scientist not in seen:
                        seen.add(scientist)
                        num_above_chance += 1
                else:
                    rank_diffs[i][j] = data[model][scientist]["win_ratio"]

            else:
                rank_diffs[i][j] = data[model][scientist][0]

    if p_vals:
        if labelled:
            p_vals = np.array(vals).reshape(num_models, num_scientists)
        else:
            p_vals = np.array([val[1] * num_scientists for val in vals]).reshape(num_models, num_scientists)
    else:
        p_vals = np.ones((num_models, num_scientists))
    models = model_order
    names = [name.replace("=", " ").replace("_", " ").replace("-", " ").replace("0001", "") for name in scientist_order]
    names = [name if "Fernando" not in name else "Fernando J Corbató" for name in names]
    #print(rank_diffs)
    maxscale = np.amax(rank_diffs, axis=0)
    minscale = np.amin(rank_diffs, axis=0)
    print(num_above_chance)
    print(num_scientists - num_above_chance)
   
    make_corr_matrix_ll(rank_diffs, models)
    hinton(p_vals, rank_diffs, names, models, maxscale=maxscale, minscale=minscale, filename=filename, include_scientist_names=include_scientist_names)

def make_corr_matrix_ll(scientists_models_data: np.ndarray, model_names) -> None:
    models = np.split(scientists_models_data, scientists_models_data.shape[0], axis=0)
    print(models)
    d = pd.DataFrame(data=scientists_models_data.T, columns=model_names)
    corr = d.corr()
    print(corr)
    print(max(corr))
    print(min(corr))
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    sns.heatmap(corr, mask=mask, cmap="summer", vmax=1.0, vmin=0.7,
            linewidths=.5, cbar_kws={"shrink": .5})
    ax = plt.gca()
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.subplots_adjust(left=0.20, bottom=0.22, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
    plt.savefig("turing-corr.png")
    plt.savefig("turing-corr.eps")



# From stackoverflow
def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v


if __name__ == "__main__":    
    #models = pickle.load(open("sbert.p", "rb"))
    #make_heatmap(models, "bla")

    #models_avg = pickle.load(open("data/pickled/abstracts/models-avg.p", "rb"))
    #make_heatmap(models_avg, filename="abstracts-avg-adjusted-new")
    #models_best = pickle.load(open("data/pickled/abstracts/models-best.p", "rb"))
    #make_heatmap(models_best, filename="abstracts-best-adjusted-new")
    #models_worst = pickle.load(open("data/pickled/abstracts/models-worst.p", "rb"))
    #make_heatmap(models_worst, filename="abstracts-worst-adjusted-new")
    models = pickle.load(open("results/full-rank/medicine.p", "rb"))
    make_heatmap(models, filename="medicine-hinton-rank", p_vals=True, labelled=True, include_scientist_names=False)
    #make_heatmap({"1NN": {"Geoffrey E. Hinton": (1.23, 0.005), "John L. Hennessy": (0.912, 0.02)}, "Exemplar": {"Geoffrey E. Hinton": (-1.23, 0.03), "John L. Hennessy": (0.912, 0.002)}})