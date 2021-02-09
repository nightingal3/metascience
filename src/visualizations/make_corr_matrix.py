import ast
import csv
from itertools import combinations
import os
from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import scipy

import pdb
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



punctuation = [".", ",", "'", "\"", ":", ";", "?", "(", ")", "[", "]"]

def closest_pair(points_list: List) -> List:
    return max((cosine_similarity(p1.reshape(1, -1), p2.reshape(1, -1))[0][0], p1, p2)
               for p1, p2 in combinations(points_list, r=2))

def smallest_n(a, n):
    return np.sort(np.partition(a, n)[:n])

def argsmallest_n(a, n):
    ret = np.argpartition(a, n)[:n]
    b = np.take(a, ret)
    return np.take(ret, np.argsort(b))

def get_knn_all_papers(vecs: List, labels: List, k: int) -> List[List]:
    closest = []
    cos_sim = cosine_similarity(vecs)

    for i, vec in enumerate(cos_sim):
        #pdb.set_trace()
        closest_inds = np.argpartition(cos_sim[i], -(k + 1))[-(k + 1):]
        closest_inds = np.delete(closest_inds, np.where(closest_inds == i))
        closest.append([labels[ind] for ind in closest_inds])

    return closest

def cleaned_title_to_original(filename: str) -> dict:
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    cleaned_to_orig = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            year, title_ch, abstract = row
            title_words = (
                "".join(
                    ch for ch in title_ch if ch not in punctuation)).lower().split(" ")
            clean_title = " ".join([lemmatizer.lemmatize(word)
                                    for word in title_words if word not in stop_words])
            cleaned_to_orig[clean_title] = title_ch
    return cleaned_to_orig


def get_distance_rank_corr(filename: str, label_mapping: dict, start_col: int = 2, label_col: int = 2) -> np.ndarray:
    with open(filename, "r") as in_file:
        print(filename)
        reader = csv.reader(in_file)
        titles = []
        abstracts = []
        labels = []

        for i, row in enumerate(reader):
            title_vec = np.array(tuple(ast.literal_eval(row[start_col])))
            abstract_vec = np.array(tuple(ast.literal_eval(row[start_col + 1])))
            titles.append(title_vec)
            abstracts.append(abstract_vec)
            if len(label_mapping.keys()) > 0:
                labels.append(label_map[row[label_col]])
        if len(titles) < 3:
            print("insufficient data")
            return np.array([])

        title_cos_sim = cosine_similarity(titles)
        #dists = scipy.spatial.distance.pdist(abstracts, "cosine")
        #closest = dists.argsort()[:3]
        #tu = np.triu_indices(title_cos_sim.shape[0], 1)
        #pairs = np.column_stack((np.take(tu[0], closest),
                         #np.take(tu[1], closest))) + 1
        closest_1 = get_knn_all_papers(titles, labels, 3)

        #ind_1 = np.argpartition(title_cos_sim, -10)[-10:]
        abstract_cos_sim = cosine_similarity(abstracts)
        #ind_2 = np.argpartition(abstract_cos_sim, -10)[-10:]
        closest_2 = get_knn_all_papers(abstracts, labels, 3)

        for i, label in enumerate(labels):
            print("=== PAPER: ", label)
            print("SBERT + Title: ")
            print(closest_1[i][0])
            print(closest_1[i][1])
            print(closest_1[i][2])

            print("\n")
            print("Fasttext + Title: ")
            print(closest_2[i][0])
            print(closest_2[i][1])
            print(closest_2[i][2])

            print("\n\n")

        title_tri = title_cos_sim[np.triu_indices(title_cos_sim.shape[0], 1)]
        abstract_tri = abstract_cos_sim[np.triu_indices(abstract_cos_sim.shape[0], 1)]
        return scipy.stats.pearsonr(title_tri, abstract_tri)

def get_knn_for_vecs(filename: str, label_mapping: dict, start_col: int = 2, multicols: bool = False, label_col: int = 2, k: int = 3) -> None:
     with open(filename, "r") as in_file:
        print(filename)
        reader = csv.reader(in_file)
        titles = []
        labels = []

        for i, row in enumerate(reader):
            if multicols:
                title_vec = np.array(tuple(ast.literal_eval(",".join(row[start_col:]))))
            else:
                title_vec = np.array(tuple(ast.literal_eval(row[start_col])))
            titles.append(title_vec)
            if len(label_mapping.keys()) > 0:
                if row[label_col] in label_map:
                    labels.append(label_map[row[label_col]])
                else:
                    labels.append(row[label_col])

        if len(titles) < 3:
            print("insufficient data")
            return np.array([])

        title_cos_sim = cosine_similarity(titles)
        closest_1 = get_knn_all_papers(titles, labels, k)

        for i, label in enumerate(labels):
            print("=== PAPER: ", label)
            for j in range(k):
                print(closest_1[i][j])
            print("\n")
           

def plot_corr_matrix(corr_matrix: np.ndarray) -> None:
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, center=0.5)
    plt.savefig("corr_matrix.png")

if __name__ == "__main__":
    label_map = cleaned_title_to_original("./data/turing_winners/abstracts/Geoffrey_E=-Hinton.csv")
    #corr = get_distance_rank_corr("./data/others/sbert-titles-vs-fasttext-titles/Yang-Xu.csv", label_map, start_col=3)
    get_knn_for_vecs("./data/turing_winners/sbert-abstracts/Geoffrey_E=-Hinton.csv", label_map, start_col=2, label_col=1, multicols=True)
    assert False
    rank_corrs = []
    for filename in os.listdir("data/turing_winners/sbert-abstracts-vs-fasttext-titles/"):
        corr = get_distance_rank_corr(f"./data/turing_winners/sbert-abstracts-vs-fasttext-titles/{filename}", {}, start_col=3)
        if len(corr) == 0:
            continue
        if corr[1] <= 0.05:
            rank_corrs.append(corr[0])
    print("MIN:", min(rank_corrs), "MAX: ",  max(rank_corrs))
    print("NUM SIGNIFICANT: ", len(rank_corrs))
