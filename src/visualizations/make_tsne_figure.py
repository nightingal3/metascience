import csv
import random
from typing import Callable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.manifold import TSNE


def select_subset(vecs_filepath: str, labels_filepath: str, out_vec_filepath: str, out_labels_filepath: str, filter: Callable[[str], bool], threshold: float = 0.5, filter_on: bool = "title") -> None:
    with open(vecs_filepath, "r") as vecs_file, open(labels_filepath, "r") as labels_file, open(out_vec_filepath, "w") as out_vecs_file, open(out_labels_filepath, "w") as out_labels_file:
        vecs_reader = csv.reader(vecs_file)
        labels_reader = csv.reader(labels_file)
        vecs_writer = csv.writer(out_vecs_file)
        labels_writer = csv.writer(out_labels_file)

        for vec_row, label_row in zip(vecs_reader, labels_reader):
            year, title = label_row
            criterion_met = (filter_on == "title" and filter(title)) or (filter_on == "year" and filter(year))     
            randomly_chosen = random.random() <= threshold  
            if criterion_met and randomly_chosen:
                vecs_writer.writerow(vec_row)
                labels_writer.writerow(label_row)


def plot_tsne(vecs: np.ndarray, years: np.ndarray, out_filename: str = "papers_tsne") -> None:
    # Hyperparams determined through some trial and error in embedding projector...
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=10, n_iter=10000, n_iter_without_progress=500)
    tsne_res = tsne.fit_transform(vecs)

    years = years.astype(np.int64)
    seaborn.scatterplot(tsne_res[:, 0], tsne_res[:, 1], hue=years, hue_norm=matplotlib.colors.Normalize(vmin=min(years), vmax=max(years)))

    plt.savefig(out_filename + ".png")
    plt.savefig(out_filename + ".eps")


if __name__ == "__main__":
    """select_subset(
        "hinton_paper_vectors.csv",
        "hinton_papers.csv",
        "hinton_vecs_selected.csv",
        "hinton_titles_selected.csv",
        lambda s: len(s.split(" ")) <= 7)"""
    data = np.genfromtxt("./data/hinton_vecs.tsv", delimiter="\t")
    years = np.genfromtxt("./data/hinton_papers.tsv", delimiter="\t", skip_header=1, usecols=(0))
    print(data, years)
    plot_tsne(data, years)
