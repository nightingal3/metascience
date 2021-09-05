import os
import pickle
from typing import List

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from make_pie_chart import make_pie_chart, get_winners
from src.models.aic_bic import calc_aic_bic_individuals, get_best_model_individual
import pdb

def collect_run_percentages(dirname: str, run_type="ll") -> List:
    exemplar = []
    for filename in os.listdir(dirname):
        if filename.endswith(".p"):
            models = pickle.load(open(os.path.join(dirname, filename), "rb"))
            if run_type == "ll":
                percentages = make_pie_chart(get_winners(models, p_vals=False, include_null=True, len_included=True), include_null=True, filename=None)
                exemplar.append(percentages["exemplar"])
            
            else:
                aic, bic = calc_aic_bic_individuals(models)
                aic_individual = get_best_model_individual(aic)
                bic_individual = get_best_model_individual(bic)
                if run_type == "aic":
                    percentages = make_pie_chart(aic_individual, include_null=True, len_included=False, filename=None)
                if run_type == "bic":
                    percentages = make_pie_chart(bic_individual, include_null=True, len_included=False, filename=None)

                exemplar.append(percentages["1NN"])

    return exemplar

def aggregate_run_percentages(dirname: str, run_type: str = "ll") -> None:
    percentages = {}
    for subdir, dirs, files in os.walk(dirname):
        if subdir == dirname:
            continue
        print(subdir)
        name_ind = subdir.rfind("/")
        percent = collect_run_percentages(subdir, run_type=run_type)
        percentages[subdir[name_ind + 1:]] = percent
    
    percentage_df = pd.DataFrame.from_dict(percentages, orient="index")
    percentage_df = percentage_df.transpose()
    return percentage_df
           
def box_chart(percentages: dict, observed: List, filename: str) -> None:
    plt.gcf().clear()
    print(percentages)
    sns.boxplot(x="variable", y="value", data=pd.melt(percentages), showfliers=False)
    #sns.boxplot(x=[2] * len(percentages["cogsci"].tolist()), y=percentages["cogsci"].tolist(), showfliers=False)
    ax = plt.gca()
    for i, y in enumerate(observed):
        ax.hlines(y, i - 0.5, i + 0.5, color="red", linestyle="dotted")
    plt.xlabel("Field")
    plt.ylabel("Predominance of exemplar model")
    plt.tight_layout()
    plt.savefig(filename + ".eps")
    plt.savefig(filename + ".png")



if __name__ == "__main__":
    # [chem, cs, econ, med, phys]
    percent_exemplar = [0.583, 0.582, 0.527, 0.709, 0.464]
    percent_exemplar_rand = [0.567, 0.406, 0.5, 0.517, 0.53]
    #percent_1NN_aic = [0.263, 0.536, 0.516, 0.288, 0.314]
    #percent_1NN_bic = [0.414, 0.594, 0.554, 0.468, 0.397]
    percent = aggregate_run_percentages("./results/shuffle-ll-new-random-sample-fixed", run_type="ll")
    #percent = collect_run_percentages("./results/summary/shuffle/physics")
    box_chart(percent, percent_exemplar_rand, "multi-test-rand-fixed")