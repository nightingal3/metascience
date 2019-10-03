import inspect
from statistics import stdev
import sys
from typing import Callable
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.predict import *
from src.models.ranking_functions import *
from src.models.chinese_restaurant import *

def get_model_cumulative_rank_diff(
    all_vecs: List,
    emergence_order: dict,
    ranking_func: Callable,
    ranking_func_name: str,
    ranking_type: str = "global",
    rank_err_filename: str = "",
) -> tuple:
    if rank_err_filename != "":
        with open(rank_err_filename, "rb") as rank_err_file:
            errs = pickle.load(rank_err_file)
            if ranking_func_name in errs:
                rank_err = errs[ranking_func_name]

    if rank_err_filename == "" or ranking_func_name not in errs:
        rank_err, _ = predict_seq(
            all_vecs,
            emergence_order,
            ranking_func,
            get_rank_score_avg,
            ranking_type=ranking_type,
        )

    p_val, rank_diff_avg, rank_diff_conf_upper, rank_diff_conf_lower, avg_rank_diff_timesteps = shuffle_test(
        n_iter=10000, target_val=rank_err, emergence_order=emergence_order
    )
    yerr = rank_diff_conf_upper - rank_diff_conf_lower

    return p_val, rank_diff_avg, yerr, rank_err


def plot_cumulative_rank_diff(models: dict) -> None:
    for i, (name, (rank_diff, err)) in enumerate(models.items()):
        plt.bar(x=i, height=rank_diff, yerr=err, label=name)
    plt.legend()
    plt.xticks([])
    plt.ylabel("Rank difference against null, 10000 trials")
    plt.savefig("rank_diff.png")
    plt.savefig("rank_diff.eps")


def get_rank_diff_over_time(
    all_vecs: List, 
    emergence_order: dict,
    ranking_func: Callable,
    ranking_func_name: str,
    ranking_type: str = "global",
    rank_err_filename: str = ""
) -> List:
    if rank_err_filename != "":
        with open(rank_err_filename, "rb") as rank_err_file:
            errs = pickle.load(rank_err_file)
            if ranking_func_name in errs:
                rank_err = errs[ranking_func_name]
    
    if rank_err_filename == "" or ranking_func_name not in errs:
        _, rank_diff_at_timestep = predict_seq(
            all_vecs,
            emergence_order,
            ranking_func,
            get_rank_score_avg,
            ranking_type=ranking_type,
        )
    
    p_val, rank_diff_avg, rank_diff_conf_upper, rank_diff_conf_lower, avg_rank_diff_timesteps = shuffle_test(
        n_iter=10, target_val=rank_err, emergence_order=emergence_order
    )
    confidence_interval_timestep = [0] * len(avg_rank_diff_timesteps)  # no error bars for now
    
    return avg_rank_diff_timesteps, confidence_interval_timestep


def plot_rank_diff_over_time(models: dict):
    fig, ax = plt.subplots()
    for model in models:
        time_data = models[model]
        rank_diffs = [i[0] for i in time_data]
        yerr = [i[1] for i in time_data]
        timesteps = range(len(rank_diffs))
        ax.errorbar(timesteps, rank_diffs, yerr=yerr, label=model, marker="o")

    plt.legend()
    plt.savefig("testing1.png")


def make_pie_chart(models: dict):
    for model in models:
        target_val = models[model]
        model_wins = shuffle_test(10000, target_val, emergence_order, return_raw_counts=True)
        plt.pie(model_wins, labels=["null", model])
        plt.legend()
        plt.savefig("{}_pie.png".format(model))
        plt.gcf().clear()

if __name__ == "__main__":
    models = {}
    bar_plot_data = {}
    line_plot_data = {}
    all_vecs = get_attested_order("data/hinton_paper_vectors.csv")
    emergence_order = get_emergence_order("data/ordered_by_date.csv")
    crp_rank = 
    name_to_func = {
        "1NN": rank_on_1NN,
        "Prototype": rank_on_prototype,
        "Progenitor": rank_on_progenitor(all_vecs[0]),
        "Exemplar": rank_on_exemplar,
        "Local": rank_on_1NN,
    }

    with open("data/model_data.p", "rb") as model_file:
        model_err = pickle.load(model_file)
        make_pie_chart(model_err)
    assert False
    """for model_name in name_to_func:
        model_type = "local" if model_name == "Local" else "global"
        _, avg_rank_diff, yerr, rank_err = get_model_cumulative_rank_diff(
            all_vecs,
            emergence_order,
            name_to_func[model_name],
            model_name,
            ranking_type=model_type,
            rank_err_filename="data/model_data.p",
        )
        # models[model_name] = rank_err
        bar_plot_data[model_name] = (avg_rank_diff, yerr)
        print(models)
    print(bar_plot_data)
    # with open("data/model_data.p", "wb") as model_f:
    # pickle.dump(models, model_f)"""

    for model_name in name_to_func:
        model_type = "local" if model_name == "Local" else "global"
        avg_rank_diff_timesteps, confidence_interval_timestep = get_rank_diff_over_time(all_vecs, emergence_order, name_to_func[model_name], model_name, ranking_type=model_type, rank_err_filename="data/model_data.p")
        line_plot_data[model_name] = list(zip(avg_rank_diff_timesteps, confidence_interval_timestep))

    #plot_cumulative_rank_diff(bar_plot_data)
    plot_rank_diff_over_time(line_plot_data)
