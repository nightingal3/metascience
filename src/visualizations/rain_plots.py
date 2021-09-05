import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from typing import List
import pickle
import numpy as np
from scipy import stats

sns.set(style="whitegrid",font_scale=1.7)
import matplotlib.collections as clt
import ptitprince as pt

from make_box_whisker import aggregate_run_percentages
from make_rank_diff_figures import get_rank_diff_and_err
from src.models.validate_embeddings import shuffle_Js, shuffle_samples_scientists_all

import pdb

def make_rain_plot(df: pd.DataFrame, out_filename: str, h_lines: List = [], plot_type: str = "shuffle", bar_field: str = "", random_field: bool = False) -> None:
    # Changing orientation
    dx="variable"; dy="value"; ort="v"; pal = "Set2"; sigma = .2
    f, ax = plt.subplots(figsize=(7, 5))

    if plot_type == "J":
        ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                        width_viol = .5, ax = ax, orient = ort)

    if plot_type == "shuffle":
        ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                        width_viol = .5, ax = ax, orient = ort, order=["physics", "chemistry", "medicine", "economics", "cs"])

    elif plot_type == "J-field":
        ax=pt.RainCloud(x = dx, y = dy, data = df, palette = pal, bw = sigma,
                                width_viol = .5, ax = ax, orient = ort, order=["Physics", "Chemistry", "Medicine", "Economics", "CS"])
    elif plot_type == "bar":
        #adding color
        pal = "Set3"
        f, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

        # regular pts
        pt.half_violinplot( x = dx, y = dy, data = df, palette = pal, bw = .2, cut = 0.,
                            scale = "area", width = .6, inner = None, orient = ort, ax=ax1)
        sns.stripplot( x = dx, y = dy, data = df, palette = pal, edgecolor = "white",
                        size = 3, jitter = 1, zorder = 0, orient = ort, ax=ax1)
        sns.boxplot( x = dx, y = dy, data = df, color = "black", width = .15, zorder = 10,\
                    showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                    showfliers=False, whiskerprops = {'linewidth':2, "zorder":10},\
                    saturation = 1, orient = ort, ax=ax1)


        # outliers
        pt.half_violinplot( x = dx, y = dy, data = df, palette = pal, bw = .2, cut = 0.,
                            scale = "area", width = .6, inner = None, orient = ort, ax=ax2)
        sns.stripplot( x = dx, y = dy, data = df, palette = pal, edgecolor = "white",
                        size = 3, jitter = 1, zorder = 0, orient = ort, ax=ax2)
        sns.boxplot( x = dx, y = dy, data = df, color = "black", width = .15, zorder = 10,\
                    showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                    showfliers=False, whiskerprops = {'linewidth':2, "zorder":10},\
                    saturation = 1, orient = ort, ax=ax2)

        # scales for two plots
        if bar_field == "physics":
            if random_field:
                ax1.set_ylim(0, 4)
                ax2.set_ylim(7, 25)
            else:
                ax1.set_ylim(0, 5)
                ax2.set_ylim(5, 25)
        elif bar_field == "medicine":
            if random_field:
                ax1.set_ylim(0, 2.5)
                ax2.set_ylim(5, 8)
            else:
                ax1.set_ylim(0, 5)
                ax2.set_ylim(7.5, 10)
        elif bar_field == "chemistry":
            if random_field:
                ax1.set_ylim(-10, -5)
                ax2.set_ylim(0, 6.5)
            else:
                ax1.set_ylim(-4, 4)
                ax2.set_ylim(4, 6.5)
        elif bar_field == "economics":
            if random_field:
                ax1.set_ylim(-1, 2)
                ax2.set_ylim(4, 6)
            else:
                ax1.set_ylim(-5, 3.5)
                ax2.set_ylim(5, 7.5)
        elif bar_field == "cs":
            if random_field:
                ax1.set_ylim(-2.5, 2)
                ax2.set_ylim(5, 12.5)
            else:
                ax1.set_ylim(0, 7)
                ax2.set_ylim(50, 200)
                ax2.set_yscale("log")
                plt.tight_layout()


        # hide spines between ax1 and ax2
        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        #ax2.set_xticklabels([])
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax1.set_ylabel("")
        ax1.xaxis.tick_bottom()

        # code for making the diagonals. From https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
        ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax = plt.gca()
    for i, y in enumerate(h_lines):
        ax.hlines(y, i - 0.5, i + 0.5, color="red", linestyle="dotted")

    if plot_type == "shuffle":
        plt.xlabel("Field")
        plt.ylabel("Predominance of exemplar model")
    elif plot_type == "bar":
        #plt.ylabel("Log-likelihood difference\nratio against null")
        f.text(0, 0.5, "Log-likelihood difference\nratio against null", ha='center', va='center', rotation='vertical')
        plt.xlabel("")
        plt.xticks(rotation=45)
    elif plot_type == "J":
        plt.yscale("log")
        plt.xticks([])
        plt.xlabel("")
        plt.ylabel("J value")
    elif plot_type == "J-field":
        plt.yscale("log")
        plt.xlabel("")
        plt.ylabel("J value")
        plt.xticks(rotation=45)

    plt.savefig(f'{out_filename}.png', bbox_inches='tight')
    plt.savefig(f'{out_filename}.eps', bbox_inches='tight')


def make_shuffle_plot(random: bool = False) -> None:
    percent_exemplar = [0.452, 0.583, 0.709, 0.514, 0.449]
    percent_exemplar_rand = [0.52, 0.583, 0.506, 0.495, 0.457]

    path = "./results/shuffle-ll-new-random-sample-fixed" if random else "./results/shuffle-ll-new"
    out_path = "rand_shuffle_rain" if random else "shuffle_rain"
    attested_percent = percent_exemplar_rand if random else percent_exemplar

    percent_df = pd.melt(aggregate_run_percentages(path, run_type="ll"))
    
    make_rain_plot(percent_df, out_path, h_lines=attested_percent)

def make_ll_bar_plot(field: str, out_filename, random: bool = False) -> None:
    model_res = {"kNN": [], "prototype": [], "progenitor": [], "exemplar": [], "local": []}
    

    field_path = f"results/full-fixed/{field}-random.p" if random else f"results/full-fixed/{field}.p"
    _, model_results = get_rank_diff_and_err(pickle.load(open(field_path, "rb")), avg=True)
    for k in model_results:
        model_res[k] = model_results[k]

    model_df = pd.melt(pd.DataFrame(model_res))

    make_rain_plot(model_df, out_filename, plot_type="bar", bar_field = field, random_field = random)

def make_J_plot_all_fields(out_filename: str, random: bool = False) -> None:
    if random:
        fields_dict, vecs_dict = pickle.load(open("./data/sbert-vecs-dict-rand.p", "rb"))
    else:
        fields_dict, _, vecs_dict = pickle.load(open("./data/sbert-vecs-dict-all.p", "rb"))
    p_val, Js, base_J = shuffle_Js(vecs_dict, num_shuffle=1000)
    df_Js = pd.melt(pd.DataFrame(Js))
    make_rain_plot(df_Js, out_filename, h_lines=[base_J], plot_type="J")

def make_J_plot_individual_fields(out_filename: str, random: bool = False) -> None:
    fields_names = {
        "data/nobel_winners/physics/abstracts-ordered": "Physics",
        "data/nobel_winners/chemistry/abstracts-ordered": "Chemistry",
        "data/nobel_winners/medicine/abstracts-ordered": "Medicine",
        "data/nobel_winners/economics/abstracts-ordered": "Economics",
        "data/turing_winners/sbert-abstracts-ordered": "CS"
    }
    fields_name_rand = {
        "data/nobel_winners/physics-random/abstracts-ordered": "Physics",
        "data/nobel_winners/chemistry-random/abstracts-ordered": "Chemistry",
        "data/nobel_winners/medicine-random/sbert-abstracts-ordered": "Medicine",
        "data/nobel_winners/economics-random/abstracts-ordered": "Economics",
        "data/nobel_winners/cs-random/abstracts-ordered": "CS"
    }
    cols_dict = {}
    if random:
        fields_dict, vecs_dict = pickle.load(open("./data/sbert-vecs-dict-rand.p", "rb"))
    else:
        fields_dict, _, vecs_dict = pickle.load(open("./data/sbert-vecs-dict-all.p", "rb"))
    Js, base_J, p_val = shuffle_samples_scientists_all(fields_dict)
    base_J_lst = []
    paths_to_name = fields_name_rand if random else fields_names
    for path_name in paths_to_name:
        cols_dict[paths_to_name[path_name]] = Js[path_name]
        base_J_lst.append(base_J[path_name])

    df = pd.melt(pd.DataFrame(cols_dict))
    make_rain_plot(df, out_filename, h_lines=base_J_lst, plot_type="J-field")

if __name__ == "__main__":
    fields = ["physics", "chemistry", "medicine", "economics", "cs"]
    for field in fields:
        make_ll_bar_plot(field, f"{field}-bar-ll", random=False)
    #assert False
    #make_shuffle_plot(random=False)
    #make_J_plot_individual_fields("J-fields-rand", random=True)
    #make_J_plot_individual_fields("J-fields", random=False)

    #make_J_plot_all_fields("J-all-rand", random=True)
    #make_J_plot_all_fields("J-all", random=False)