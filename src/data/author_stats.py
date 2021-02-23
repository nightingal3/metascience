import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from scipy.stats import ttest_ind
import os
import pdb

def get_authors_above_chance(model_dict: dict, p_vals: bool = False) -> List:
    scientists_to_models = {k: (-float("inf"), None) for k in model_dict['1NN']}

    for model in model_dict:
        print("MODEL: ", model)
        for scientist in model_dict[model]:
            print("SCIENTIST: ", scientist)
            print(model_dict[model][scientist])
            if p_vals and model_dict[model][scientist][1] * len(scientists_to_models) > 0.05:
                continue
            if p_vals:
                val = model_dict[model][scientist][0]
            else:
                val = model_dict[model][scientist][0]
            print(val)
            if val > scientists_to_models[scientist][0]:
                scientists_to_models[scientist] = (val, model)
    print(scientists_to_models)
    authors_above_chance = [k for k, v in scientists_to_models.items() if v[0] > 0]
    authors_below_chance = [k for k, v in scientists_to_models.items() if v[0] <= 0]

    return authors_above_chance, authors_below_chance


def get_histogram_data(data_filename: str, authors_above_chance: List, authors_below_chance: List, col: int = 2) -> tuple:
    num_authors_above = []
    num_authors_below = []
    for author in authors_above_chance:
        in_filename = os.path.join(data_filename, author + ".csv")
        num_authors_above.append(get_avg_author_number(in_filename, col=col))
    for author in authors_below_chance:
        in_filename = os.path.join(data_filename, author + ".csv")
        num_authors_below.append(get_avg_author_number(in_filename, col=col))

    return num_authors_above, num_authors_below

def make_histogram(dist1: List, dist2: List) -> None:
    sns.distplot(dist1, color="skyblue", label="Predicted above chance", norm_hist=True)
    sns.distplot(dist2, color="red", label="Predicted below chance", norm_hist=True)
    plt.legend()
    print(ttest_ind(dist1, dist2))
    plt.savefig("above-below-sbert.png")
    plt.savefig("above-below-sbert.eps")
    plt.show()

def get_avg_author_number(in_filename: str, col: int = 2) -> None:
    df = pd.read_csv(in_filename, header=None)
    author_num = df[col].mean(axis=0)
    if author_num != author_num:
        pdb.set_trace()
    return author_num


if __name__ == "__main__":
    models = pickle.load(open("results/full-2/physics.p", "rb"))
    del models["Null"]
    above_chance, below_chance = get_authors_above_chance(models)
    #a_c, b_c = get_histogram_data("data/turing_winners/vecs-abstracts-w-labels/labels-and-authors", above_chance, below_chance, col=3)
    print("ABOVE CHANCE:", len(above_chance))
    print("BELOW CHANCE: ", len(below_chance))
    #make_histogram(a_c, b_c)
    #get_avg_author_number("data/turing_winners/vecs-abstracts-w-labels/labels-and-authors/Adi-Shamir.csv")


