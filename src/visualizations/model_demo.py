import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List
import pdb
from scipy.spatial import distance
from sklearn.manifold import TSNE

from src.models.predict import rank_on_1NN, rank_on_exemplar, rank_on_progenitor, rank_on_prototype, make_rank_on_knn, get_prototype, get_emergence_order, get_attested_order, get_probability_score

def gen_random_2d_data(n: int) -> np.ndarray:
    return np.random.rand(n, 2)


def link_with_model(data: np.ndarray, model: Callable, model_name: str, dist="euclidean") -> np.ndarray:
    order = [0]
    emerged = [data[0]]
    tails = []
    available_vecs = np.copy(data[1:])

    while len(order) < data.shape[0]:
        last_papers = emerged
        if model_name == "Local":
            last_papers = [emerged[-1]]
            assert len(last_papers) == 1
        preds = model(last_papers, available_vecs, dist=dist)
        if model_name == "1NN":
            closest_ind = np.argpartition(distance.cdist(np.array([preds[0]]), np.asarray(emerged), metric=dist), 0)[0][0]
            tails.append(closest_ind)
        elif model_name == "Prototype":
            proto = get_prototype(last_papers)
            closest_ind = np.argpartition(distance.cdist(np.array([proto]), np.asarray(emerged), metric=dist), 0)[0][0]
            tails.append(closest_ind)
        elif model_name == "Progenitor":
            tails.append(0)
        elif model_name == "Exemplar":
            proto = get_prototype(emerged)
            tails.append(proto)

        emerged.append(preds[0])
        #print(preds[0])
        orig_ind = np.where((data == preds[0]).all(axis=1))[0][0]
        new_ind = np.where((available_vecs == preds[0]).all(axis=1))[0][0]
        order.append(orig_ind)
        available_vecs = np.delete(available_vecs, new_ind, axis=0)

    if model_name == "Local":
        tails = range(0, len(order) - 1)

    return order, tails

def plot_links(data: np.ndarray, order: List, tails: List, model_type: str, filename: str) -> None:
    order_sort = np.argsort(order)
    ordered_data = [None for _ in range(data.shape[0])]
    plt.scatter(data[0, 0], data[0, 1], facecolors="magenta", edgecolors="magenta", s=200, linewidths=5)
    plt.scatter(data[1:, 0], data[1:, 1], facecolors="none", edgecolors="blue", s=200, linewidths=5)

    for i, ind in enumerate(order_sort):
        
        plt.annotate(ind, (data[i, 0] + 0.01, data[i, 1] + 0.01), color="red", fontsize=40)
        print(f"ind: {ind}, coords: {data[i]}")
        ordered_data[ind] = data[i]
        #prev_pt = data[tails[i - 1]]
        #plt.arrow(prev_pt[0], prev_pt[1], data[i, 0] - prev_pt[0], data[i, 1] -  prev_pt[1])
    #pdb.set_trace()
    #plt.arrow(ordered_data[tails[0]][0], ordered_data[tails[0]][1], ordered_data[1][0] - ordered_data[tails[0]][0], ordered_data[1][1] - ordered_data[tails[0]][1], head_width=0.012, head_length=0.015, length_includes_head=True, color="black")
    #print("x: ", ordered_data[tails[0]][0])
    #print("y: ", ordered_data[tails[0]][1])
    #pdb.set_trace()
    for i, curr_pt in enumerate(tails):
        #print("==iteration==")
        if model_type == "Exemplar":
            prev_pt = tails[i]
        else:
            prev_pt = ordered_data[tails[i]]
        curr_pt = ordered_data[i + 1]
        #print("prev: ", tails[i])
        #print(ordered_data[tails[i]])
        #print("next: ", i + 1)
        #print(ordered_data[i + 1])
        plt.arrow(prev_pt[0], prev_pt[1], curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1], head_width=0.04, head_length=0.03, length_includes_head=True, color="black")
    #print("data: ", data)
    #print("ordered data: ", ordered_data)
    #print("order sort: ", order_sort)
    plt.axis("off")
    plt.savefig(f"{filename}-chain-demo.png")
    plt.savefig(f"{filename}-chain-demo.eps")
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()

if __name__ == "__main__":
    """all_vecs = get_attested_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", vecs_col=2)
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate=200,
        n_iter=10000,
        n_iter_without_progress=1000,
    )
    
    tsne_res = tsne.fit_transform(all_vecs)

    labels = get_attested_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", vecs_col=1, label=True)
    emergence_order = get_emergence_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final-ordered.csv", vecs_col=2)
    _, ranks, tails = get_probability_score(emergence_order, all_vecs, labels, rank_on_1NN, ranking_type="global", carry_error=True)
    inds = [all_vecs.index(list(vec)) for vec in ranks]
    plot_links(tsne_res, np.asarray(inds), np.asarray(tails), "1NN-geoff", "1NN-geoff")
    assert False"""
    data = gen_random_2d_data(10)
    #data = get_attested_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", vecs_col=2)

    links_nn, tails_nn = link_with_model(data, rank_on_1NN, "1NN")
    links_progen, tails_progen = link_with_model(data, rank_on_progenitor(data[0]), "Progenitor")
    links_proto, tails_proto = link_with_model(data, rank_on_prototype, "Prototype")
    links_exemplar, tails_exemplar = link_with_model(data, rank_on_exemplar, "Exemplar")
    links_local, tails_local = link_with_model(data, rank_on_1NN, "Local")

    plot_links(data, links_nn, tails_nn, "1NN", "1NN")
    plot_links(data, links_progen, tails_progen, "Progenitor", "Progenitor")
    plot_links(data, links_proto, tails_proto, "Prototype", "Prototype")
    plot_links(data, links_exemplar, tails_exemplar, "Exemplar", "Exemplar")
    plot_links(data, links_local, tails_local, "Local", "Local")
