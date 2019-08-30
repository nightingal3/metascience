import csv
import random
from typing import Callable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def plot_tsne(vecs: np.ndarray) -> None:
    # Hyperparams determined through some trial and error in embedding projector...
    tsne = TSNE(n_components=300, perplexity=5, learning_rate=20, n_iter=10000, n_iter_without_progress=500)
    tsne_res = tsne.fit_transform(vecs)

if __name__ == "__main__":
    select_subset(
        "hinton_paper_vectors.csv",
        "hinton_papers.csv",
        "hinton_vecs_selected.csv",
        "hinton_titles_selected.csv",
        lambda s: len(s.split(" ")) <= 7)
