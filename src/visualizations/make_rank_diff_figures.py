import inspect
from statistics import stdev
import sys
from typing import Callable, List, Generic
import pickle
import os

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
    vecs_filename: str,
    order_filename: str,
    ranking_model: Generic = None,
    ranking_type: str = "global",
    rank_err_filename: str = "",
) -> tuple:
    if rank_err_filename != "":
        with open(rank_err_filename, "rb") as rank_err_file:
            errs = pickle.load(rank_err_file)
            if ranking_func_name in errs:
                rank_err = errs[ranking_func_name]

    if ranking_type == "CRP":
        rank_err, _ = predict_seq(
            all_vecs,
            emergence_order,
            None,
            get_rank_score_avg,
            ranking_model=ranking_model,
            ranking_type=ranking_type
        )
    if rank_err_filename == "" or ranking_func_name not in errs:
        rank_err, _ = predict_seq(
            all_vecs,
            emergence_order,
            ranking_func,
            get_rank_score_avg,
            ranking_type=ranking_type,
        )
    print(f"==={vecs_filename}===")
    p_val, rank_diff_avg, rank_diff_conf_upper, rank_diff_conf_lower, avg_rank_diff_timesteps = shuffle_test(
        n_iter=10000, target_val=rank_err, emergence_order=emergence_order, vecs_filename=vecs_filename, order_filename=order_filename
    )
    yerr = rank_diff_conf_upper - rank_diff_conf_lower

    return p_val, rank_diff_avg, yerr, rank_err


def plot_cumulative_rank_diff(models: dict, out_name: str) -> None:
    for i, (name, (rank_diff, err, _)) in enumerate(models.items()):
        plt.bar(x=i, height=rank_diff, yerr=err, label=name)
    plt.legend()
    plt.xticks([])
    plt.ylabel("Rank difference against null, 10000 trials")
    plt.savefig(out_name + "_rank_diff.png")
    plt.savefig(out_name + "_rank_diff.eps")
    plt.gcf().clear()


def get_rank_diff_over_time(
    all_vecs: List, 
    emergence_order: dict,
    ranking_func: Callable,
    ranking_func_name: str,
    ranking_model: Generic = None,
    ranking_type: str = "global",
    rank_err_filename: str = ""
) -> List:
    if rank_err_filename != "":
        with open(rank_err_filename, "rb") as rank_err_file:
            errs = pickle.load(rank_err_file)
            if ranking_func_name in errs:
                rank_err = errs[ranking_func_name]

    if ranking_type == "CRP":
        rank_err, _ = predict_seq(
            all_vecs,
            emergence_order,
            None,
            get_rank_score_avg,
            ranking_model=ranking_model,
            ranking_type=ranking_type
        )
    
    elif rank_err_filename == "" or ranking_func_name not in errs:
        rank_err, rank_diff_at_timestep = predict_seq(
            all_vecs,
            emergence_order,
            ranking_func,
            get_rank_score_avg,
            ranking_func_obj,
            ranking_type=ranking_type,
        )
    
    p_val, rank_diff_avg, rank_diff_conf_upper, rank_diff_conf_lower, avg_rank_diff_timesteps = shuffle_test(
        n_iter=10, target_val=rank_err, emergence_order=emergence_order
    )
    confidence_interval_timestep = [0] * len(avg_rank_diff_timesteps)  # no error bars for now
    
    return avg_rank_diff_timesteps, confidence_interval_timestep


def plot_rank_diff_over_time(models: dict, out_name: str):
    fig, ax = plt.subplots()
    for model in models:
        time_data = models[model]
        rank_diffs = [i[0] for i in time_data]
        yerr = [i[1] for i in time_data]
        timesteps = range(len(rank_diffs))
        ax.errorbar(timesteps, rank_diffs, yerr=yerr, label=model, marker="o")

    plt.legend()
    plt.savefig(out_name + ".png")


def make_pie_chart(models: dict):
    for model in models:
        target_val = models[model]
        model_wins = shuffle_test(10000, target_val, emergence_order, return_raw_counts=True)
        plt.pie(model_wins, labels=["null", model])
        plt.legend()
        plt.savefig("{}_pie.png".format(model))
        plt.gcf().clear()


def run_all(vecs_path: str, order_path: str) -> None:
    for filename in os.listdir(vecs_path):
        models = {}
        bar_plot_data = {}
        line_plot_data = {}
        if filename.endswith(".csv"):
            print(filename[:-4])
            all_vecs = get_attested_order(vecs_path + filename)
            emergence_order = get_emergence_order(order_path + filename)
            name_to_func = {
                "1NN": rank_on_1NN,
                "Prototype": rank_on_prototype,
                "Progenitor": rank_on_progenitor(all_vecs[0]),
                "Exemplar": rank_on_exemplar,
                "Local": rank_on_1NN,
                #"CRP": crp_model.rank_on_clusters
            }

            for model_name in name_to_func:
                model_type = "local" if model_name == "Local" else "global"
                pval, avg_rank_diff, yerr, rank_err = get_model_cumulative_rank_diff(
                    all_vecs,
                    emergence_order,
                    name_to_func[model_name],
                    model_name,
                    vecs_filename=vecs_path + filename,
                    order_filename=order_path + filename,
                    ranking_type=model_type,
                    #rank_err_filename="data/model_data.p",
                )
                pval = pval 
                models[model_name] = rank_err
                bar_plot_data[model_name] = (avg_rank_diff, yerr, pval)

            with open(f"data/{filename[:-4]}.p", "wb") as model_f:
                pickle.dump(models, model_f)

            plot_cumulative_rank_diff(bar_plot_data, filename[:-4])
            assert False
            """except:
                print(f"Error on {filename}")
                continue"""


if __name__ == "__main__":
    #run_all("data/turing_winners/vecs/", "data/turing_winners/ordered/")
    vecs_path = "data/turing_winners/vecs/"
    for filename in os.listdir(vecs_path):
        if filename.endswith(".csv"):
            models = {}
            bar_plot_data = {}
            line_plot_data = {}
            vecs_path = "data/turing_winners/vecs/"
            vecs_filename = f"{vecs_path}{filename}"
            order_path = "data/turing_winners/ordered/"
            order_filename = f"{order_path}{filename}"
            full_name = filename[:-4]
            
            all_vecs = get_attested_order(vecs_filename)
            emergence_order = get_emergence_order(order_filename)
            #crp_rank = create_crp([[all_vecs[0]]], 0.1)
            #get_rank_diff_over_time(all_vecs, emergence_order, crp_rank, "CRP")
            #crp_model = CRP(0.01, all_vecs[0])
            #get_rank_diff_over_time(all_vecs, emergence_order, crp_model, ranking_func_name="CRP", ranking_type="CRP")
            #get_model_cumulative_rank_diff(all_vecs, emergence_order, None, ranking_model=crp_model, ranking_func_name="CRP", ranking_type="CRP")
            name_to_func = {
                "1NN": rank_on_1NN,
                "Prototype": rank_on_prototype,
                "Progenitor": rank_on_progenitor(all_vecs[0]),
                "Exemplar": rank_on_exemplar,
                "Local": rank_on_1NN,
                #"CRP": crp_model.rank_on_clusters
            }

            for model_name in name_to_func:
                model_type = "local" if model_name == "Local" else "global"
                pval, avg_rank_diff, yerr, rank_err = get_model_cumulative_rank_diff(
                    all_vecs,
                    emergence_order,
                    name_to_func[model_name],
                    model_name,
                    ranking_type=model_type,
                    vecs_filename=vecs_filename,
                    order_filename=order_filename
                    #rank_err_filename="data/model_data.p",
                )
                pval = pval
                models[model_name] = (rank_err, pval)
                bar_plot_data[model_name] = (avg_rank_diff, yerr, pval)
                print(models)
            print(bar_plot_data)
            with open(f"data/{full_name}.p", "wb") as model_f:
                pickle.dump(models, model_f)

            """for model_name in name_to_func:
                model_type = "local" if model_name == "Local" else "global"
                avg_rank_diff_timesteps, confidence_interval_timestep = get_rank_diff_over_time(all_vecs, emergence_order, name_to_func[model_name], model_name, ranking_type=model_type, rank_err_filename="data/model_data.p")
                line_plot_data[model_name] = list(zip(avg_rank_diff_timesteps, confidence_interval_timestep))
            """
            plot_cumulative_rank_diff(bar_plot_data, f"{full_name}")
    #plot_rank_diff_over_time(line_plot_data, "John-McCarthy")"""
