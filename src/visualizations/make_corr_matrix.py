import ast
import csv
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import scipy

def get_distance_rank_corr(filename: str) -> np.ndarray:
    with open(filename, "r") as in_file:
        print(filename)
        reader = csv.reader(in_file)
        titles = []
        abstracts = []

        for row in reader:
            title_vec = np.array(tuple(ast.literal_eval(row[2])))
            abstract_vec = np.array(tuple(ast.literal_eval(row[3])))
            titles.append(title_vec)
            abstracts.append(abstract_vec)
        if len(titles) < 3:
            print("insufficient data")
            return np.array([])
        title_cos_sim = cosine_similarity(titles)
        abstract_cos_sim = cosine_similarity(abstracts)

        title_tri = title_cos_sim[np.triu_indices(title_cos_sim.shape[0], 1)]
        abstract_tri = abstract_cos_sim[np.triu_indices(abstract_cos_sim.shape[0], 1)]
        return scipy.stats.pearsonr(title_tri, abstract_tri)


def plot_corr_matrix(corr_matrix: np.ndarray) -> None:
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, center=0.5)
    plt.savefig("corr_matrix.png")

if __name__ == "__main__":
    for filename in os.listdir("./data/turing_winners/vecs-abstracts"):
        corr = get_distance_rank_corr(f"./data/turing_winners/vecs-abstracts/{filename}")
        print(corr)
