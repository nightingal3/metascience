from statistics import stdev
import sys

import matplotlib.pyplot as plt

from src.models.predict import *

def get_model_cumulative_rank_diff():
    all_vecs = get_attested_order("data/hinton_paper_vectors.csv")
    emergence_order = get_emergence_order("data/ordered_by_date.csv")

    rank_error = predict_seq(all_vecs, emergence_order, rank_on_1NN, get_rank_score_avg, ranking_type="global")

    error, rank_diff_conf_upper, rank_diff_conf_lower = shuffle_test(n_iter=100, target_val=rank_error, emergence_order=emergence_order)
    print(error)
    print(rank_diff_conf_upper)
    print(rank_diff_conf_lower)

def plot_cumulative_rank_diff(models: dict) -> None:
    for i, (name, (rank_diff, err)) in enumerate(models.items()):
        plt.bar(x=i, height=rank_diff, yerr=err, label=name)
    plt.legend()
    plt.xticks([])
    plt.ylabel("Rank difference against null, 10000 trials")
    plt.show()

def plot_rank_diff_over_time(models: dict):
    assert NotImplementedError

if __name__ == "__main__":
    models = {}
    get_model_cumulative_rank_diff()