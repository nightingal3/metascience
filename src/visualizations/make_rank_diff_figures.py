import inspect
from statistics import stdev
import sys
from typing import Callable, List, Generic
import pickle
import os
import statistics

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import matplotlib.cm as cm
import scipy.stats as st
import matplotlib.patches as mpatches

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

def get_model_cumulative_rank_diff_multi(
    all_vecs: List,
    emergence_order: dict,
    target_val_list: List, 
    vecs_filename: str, 
    order_filename: str) -> tuple:
        p_vals, rank_diff_avgs, rank_diff_conf_uppers, rank_diff_conf_lowers, avg_rank_diff_timesteps_multi = shuffle_test_multi(
            n_iter=10000, target_val_list=target_val_list, emergence_order=emergence_order, vecs_filename=vecs_filename, order_filename=order_filename
        )
        yerrs = [rank_diff_conf[0] - rank_diff_conf[1] for rank_diff_conf in zip(rank_diff_conf_uppers, rank_diff_conf_lowers)]
        return p_vals, rank_diff_avgs, yerrs


def get_rank_diff_and_err(models: dict, avg: bool = True) -> dict:
    out_models = {}
    for model in models:
        rank_diffs = []
        for scientist in models[model]:
            rank_diffs.append(models[model][scientist])
        if avg: 
            avg_rank_diff = sum(rank_diffs) / len(rank_diffs)
        ci = 1.96 * (statistics.stdev(rank_diffs) / np.sqrt(len(rank_diffs)))
        if avg:
            out_models[model] = (avg_rank_diff, 2 * ci)
        else:
            out_models[model] = (rank_diffs, 2 * ci)
        t_stat, p_val = st.ttest_1samp(rank_diffs, 0)
        print(f"{model}: ({t_stat}, {p_val})")
    return out_models


def plot_cumulative_rank_diff(models: dict, out_name: str) -> None:
    order = {
        "1NN": 0,
        "2NN": 1,
        "3NN": 2,
        "4NN": 3,
        "5NN": 4, 
        "Progenitor": 5,
        "Prototype": 6,
        "Exemplar": 7,
        "Local": 8,
        "Null": 9
    }
    for i, (name, (rank_diff, err)) in enumerate(models.items()):
        plt.bar(x=i, height=rank_diff, yerr=err, label=name, facecolor=cm.Set3.colors[order[name]])
    plt.legend(loc='upper center', bbox_to_anchor=(1.4, 0.9), fontsize=14)
    plt.xticks([])
    plt.ylabel("Log-likelihood difference ratio\nagainst null", fontsize=18)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(out_name + "_rank_diff.png")
    plt.savefig(out_name + "_rank_diff.eps")
    plt.gcf().clear()


def violin_plot(models: dict, out_name: str) -> None:
    order = {
        "1NN": 0,
        "2NN": 1,
        "3NN": 2,
        "4NN": 3,
        "5NN": 4, 
        "Progenitor": 5,
        "Prototype": 6,
        "Exemplar": 7,
        "Local": 8,
    }

    rank_diffs = [model_info[1][0] for model_info in models.items()]
    names = [model_info[0] for model_info in models.items()]
    violin_parts = plt.violinplot(rank_diffs, showmedians=True)

    patches = []
    for i in range(9):
        patches.append(mpatches.Patch(color=cm.Set3.colors[i]))

    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(cm.Set3.colors[i])
        pc.set_edgecolor(cm.Set3.colors[i])

    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = violin_parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    plt.legend(patches, names, loc='upper center', bbox_to_anchor=(1.4, 0.9), fontsize=14)
    plt.xticks([])
    plt.ylabel("Log-likelihood difference ratio\nagainst null", fontsize=18)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(out_name + "_rank_diff_v.png")
    plt.savefig(out_name + "_rank_diff_v.eps")
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
        n_iter=10000, target_val=rank_err, emergence_order=emergence_order
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



if __name__ == "__main__":
    models = get_rank_diff_and_err(pickle.load(open("model-LL.p", "rb")), avg=True)
    plot_cumulative_rank_diff(models, "models-LL-bar")
    assert False
    vecs_path = "data/turing_winners/vecs-abstracts-ordered"
    for filename in os.listdir(vecs_path):
        if filename.endswith(".csv"):
            if "Frederick" not in filename:
                continue
            vecs_filename = f"{vecs_path}/{filename}"
            order_filename = f"{vecs_path}/{filename}"
            all_vecs_abs = get_attested_order(vecs_filename)
            emergence_order_abs = get_emergence_order(order_filename)
            if "Frederick" in filename:
                name_to_func = {
                            "1NN": rank_on_1NN,
                            "2NN": make_rank_on_knn(2),
                            "3NN": make_rank_on_knn(3),
                            "4NN": make_rank_on_knn(4),
                            "5NN": make_rank_on_knn(5),
                            "Prototype": rank_on_prototype,
                            "Progenitor": rank_on_progenitor(all_vecs_abs[0]),
                            "Exemplar": rank_on_exemplar,
                            "Local": rank_on_1NN
                            #"CRP": crp_model.rank_on_clusters
                        }
            else:
                name_to_func = {
                    "2NN": make_rank_on_knn(2),
                    "3NN": make_rank_on_knn(3),
                    "4NN": make_rank_on_knn(4),
                    "5NN": make_rank_on_knn(5)
                }
            models = {}
            models_full_info = {}
            rank_errs = []
            avg_cumulative_errs = []
            model_order = {}

            for i, model in enumerate(name_to_func):
                model_type = "local" if model == "Local" else "global"
                error, rank_diff_per_timestep = predict_seq(
                    all_vecs_abs,
                    emergence_order_abs,
                    name_to_func[model],
                    get_rank_score_avg,
                    ranking_type=model_type,
                )
                avg_cumulative_errs.append(error)
                model_order[i] = model

            pvals, rank_diff_avgs, yerrs = get_model_cumulative_rank_diff_multi(all_vecs_abs, emergence_order_abs, avg_cumulative_errs, vecs_filename, order_filename)
            for i in range(len(avg_cumulative_errs)):
                models[model_order[i]] = (rank_diff_avgs[i], yerrs[i])
                models_full_info[model_order[i]] = (rank_diff_avgs[i], yerrs[i], pvals[i])
            if "Adi-Shamir" in filename:
                old_models = {}
            else:
                with open(f"data/pickled/abstracts/avg/{filename[:-4]}-avg-abs.p", "rb") as p_file:
                    old_models = pickle.load(p_file)
            old_models.update(models_full_info)
            with open(f"data/pickled/abstracts/avg/{filename[:-4]}-avg-abs.p", "wb") as p_file:
                pickle.dump(old_models, p_file)
            
            plot_cumulative_rank_diff(models, out_name=f"results/{filename[:-4]}-abstract-avg-NN")
    assert False

    vecs_path = "data/turing_winners/vecs/"
    for filename in os.listdir(vecs_path):
        if filename.endswith(".csv"):
            print(f"==={filename}===")
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
