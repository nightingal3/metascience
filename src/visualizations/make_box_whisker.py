import os
import pickle
from typing import List

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from make_pie_chart import make_pie_chart, get_winners
import pdb

def collect_run_percentages(dirname: str) -> List:
    exemplar = []
    for filename in os.listdir(dirname):
        if filename.endswith(".p"):
            models = pickle.load(open(os.path.join(dirname, filename), "rb"))
            percentages = make_pie_chart(get_winners(models, p_vals=False, include_null=True), include_null=False, filename=None)
            exemplar.append(percentages["Exemplar"])
    
    return exemplar

def aggregate_run_percentages(dirname: str) -> None:
    percentages = {}
    for subdir, dirs, files in os.walk(dirname):
        if subdir == dirname:
            continue
        print(subdir)
        name_ind = subdir.rfind("/")
        percent = collect_run_percentages(subdir)
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
    fold_2_percent = [0.842, 0.444, 0.657, 0.936, 0.765]
    fold_1_percent =  [0.847, 0.355, 0.541, 0.873, 0.741]
    percent = aggregate_run_percentages("./results/shuffle/fold-1")
    #percent = collect_run_percentages("./results/summary/shuffle/physics")
    box_chart(percent, fold_1_percent, "multi-test-1")