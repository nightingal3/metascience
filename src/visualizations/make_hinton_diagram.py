from typing import List
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.hinton import hinton

def find_rank_diff(filenames: List) -> dict:
    raise NotImplementedError

# data format: {model: {name: (rank diff, p value)}}
def make_heatmap(data: dict, filename: str) -> None:
    vals = list(NestedDictValues(data))
    num_models = len(data)
    num_scientists = len(data["1NN"])
    rank_diffs = np.array([val[0] for val in vals]).reshape(num_models, num_scientists)
    p_vals = np.array([val[1] * num_scientists for val in vals]).reshape(num_models, num_scientists)
    models = list(data.keys())
    names = [name.replace("=", " ").replace("_", " ").replace("-", " ").replace("0001", "") for name in list(data[models[0]].keys())]
    
    hinton(p_vals, rank_diffs, names, models, maxscale=max([val[0] for val in vals]), filename=filename)


# From stackoverflow
def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v


if __name__ == "__main__":
    models_avg = pickle.load(open("data/pickled/abstracts/models-avg.p", "rb"))
    make_heatmap(models_avg, filename="abstracts-avg-adjusted-new")
    models_best = pickle.load(open("data/pickled/abstracts/models-best.p", "rb"))
    make_heatmap(models_best, filename="abstracts-best-adjusted-new")
    models_worst = pickle.load(open("data/pickled/abstracts/models-worst.p", "rb"))
    make_heatmap(models_worst, filename="abstracts-worst-adjusted-new")

    #make_heatmap({"1NN": {"Geoffrey E. Hinton": (1.23, 0.005), "John L. Hennessy": (0.912, 0.02)}, "Exemplar": {"Geoffrey E. Hinton": (-1.23, 0.03), "John L. Hennessy": (0.912, 0.002)}})