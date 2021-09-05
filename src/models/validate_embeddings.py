from random import sample, shuffle
import os
import pdb
import pickle
from typing import List

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from src.models.predict import get_attested_order, get_prototype

def shuffle_samples(vecs_dict_full: dict, num_samples: int = 20, num_scientists: int = 10, num_papers: int = 20, num_shuffles: int = 1000) -> List:
    """Take num_samples samples of num_scientists each with num_papers and return a list of p-values of a shuffle test (with num_shuffle shuffles)

    Args:
        vecs_dict_full (dict): [description]
        num_samples (int, optional): [description]. Defaults to 20.

    Returns:
        List: [description]
    """
    p_vals = []
    for _ in range(num_samples):
        vecs_dict = sample_random_scientists_and_papers(vecs_dict_full, num_scientists, num_papers)

        p_val, _, _ = shuffle_Js(vecs_dict, num_shuffles)

        p_vals.append(p_val)

    return p_vals

def shuffle_samples_scientists(vecs_dict_full: dict, num_samples: int = 20, num_scientists: int = 10, num_papers: int = 20, num_shuffles: int = 1000) -> dict:
    p_vals: {}

    for field in vecs_dict_full:
        p_vals[field] = []
        for _ in range(num_samples):
            vecs_dict = sample_random_papers_by_scientist(vecs_dict_full[field], num_scientists, num_papers)

            p_val, _, _ = shuffle_Js(vecs_dict, num_shuffles)

            p_vals[field].append(p_val)

    return p_vals

def shuffle_samples_scientists_all(vecs_dict_full: dict) -> tuple:
    field_Js = {}
    base_Js = {}
    p_vals = {}
    for field in vecs_dict_full:
        scientist_dict = sample_random_papers_by_scientist(vecs_dict_full, field, select_all=True)

        p_val, Js, base_J = shuffle_Js(scientist_dict)
        field_Js[field] = Js
        base_Js[field] = base_J
        p_vals[field] = p_val

    return field_Js, base_Js, p_vals

def shuffle_Js(vecs_dict: dict, num_shuffle: int = 1000) -> tuple:
    """Run a shuffle test of the statistic J (validate function), shuffling scientists between domains or papers between scientists.

    Args:
        num_shuffle (int, optional): Number of trials. Defaults to 100.

    Returns:
        float: p-value
    """
    base_J = validate(vecs_dict)
    Js = []

    for _ in range(num_shuffle):
        shuffled_dict = shuffle_fields(vecs_dict)
        J = validate(shuffled_dict)
        Js.append(J)

    larger_Js = [J for J in Js if J >= base_J]

    return len(larger_Js)/len(Js), Js, base_J

def shuffle_fields(vecs_dict: dict) -> dict:
    """Shuffle scientists between fields, or papers between scientists.

    Args:
        vecs_dict (dict): {field: [[scientist 1 paper list], [scientist 2 paper list...]]}

    Returns:
        dict: same format as original dict, but with scientists shuffled
    """
    all_vecs = flatten_dict(vecs_dict)
    shuffle(all_vecs)
    num_used = 0

    shuffled_dict = {}
    for field in vecs_dict:
        num_in_field = len(vecs_dict[field])
        shuffled_dict[field] = all_vecs[num_used:num_used + num_in_field]
        num_used += num_in_field

    return shuffled_dict

def validate(vecs_dict: dict) -> float:
    """[summary]

    Args:
        vecs_dict (dict): [description]

    Returns:
        tuple: [description]
    """
    #pdb.set_trace()
    mean_vec = get_prototype(flatten_dict(vecs_dict))
    category_means = []
    V_w = 0
    for category in vecs_dict:
        #pdb.set_trace()
        to_flatten = {category: vecs_dict[category]}
        flattened_vecs = flatten_dict(to_flatten)
        category_mean_vec = get_prototype(flattened_vecs)
        category_means.append(category_mean_vec)
        V_w += calc_variance(category_mean_vec, flattened_vecs)

    V_b = calc_variance(mean_vec, np.array(category_means))

    return V_b / V_w

def flatten_dict(vecs_dict: dict) -> List:
    all_vecs = []
    for field in vecs_dict:
        for scientist_vecs in vecs_dict[field]:
            all_vecs.append(scientist_vecs)

    return all_vecs

def calc_variance(mean_vec: np.ndarray, query_vec_list: np.ndarray) -> float:
    mean_vec = np.expand_dims(mean_vec, axis=0)
    return cdist(query_vec_list, mean_vec, metric="sqeuclidean").sum()

def get_vecs_field(vecs_path_list: List, num_to_select: int = 10, select_all: bool = False) -> dict:
    fields_dict = {}
    for vecs_path in vecs_path_list:
        fields_dict[vecs_path] = []
        for i, filename in enumerate(os.listdir(vecs_path)): 
            vecs_filename = os.path.join(vecs_path, filename)
            all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
            if len(all_vecs) < num_to_select and not select_all:
                continue
            fields_dict[vecs_path].append(all_vecs)
            scientist_name = filename[:-4]
            print(scientist_name)

    papers_sample = sample_random_scientists_and_papers(fields_dict, num_to_select, num_to_select, select_all=select_all)
            
    return fields_dict, papers_sample

def sample_random_scientists_and_papers(fields_dict: dict, num_scientists: int = 10, num_papers: int = 10, select_all: bool = False) -> dict:
    scientists_sample = {}
    for field in fields_dict:
        if select_all:
            scientists_sample[field] = fields_dict[field]
        else:
            scientists_sample[field] = sample(fields_dict[field], num_scientists)

    papers_sample = {}
    for field in scientists_sample:
        if field not in papers_sample:
            papers_sample[field] = []
        for scientist_papers in scientists_sample[field]:
            if select_all:
                papers_sample[field].extend(scientist_papers)
            else:
                try:
                    paper_sample = sample(scientist_papers, num_papers)
                except ValueError: # scientist doesn't have enough papers - should account for this in prev loop
                    continue
                papers_sample[field] = paper_sample

    return papers_sample

def sample_random_papers_by_scientist(fields_dict_full: dict, field_path: str, num_scientists: int = 20, num_papers: int = 10, select_all: bool = False) -> dict:
    scientists_dict = {}
    scientist_sample = sample(range(len(fields_dict_full[field_path])), num_scientists)
    for i, scientist_papers in enumerate(fields_dict_full[field_path]):
        if i not in scientist_sample:
            continue
        if select_all:
            scientists_dict[i] = scientist_papers
        else:
            scientists_dict[i] = sample(scientist_papers, num_papers)

    return scientists_dict


def plot_Js_box_whisker(Js: list, base_J: int, out_filename: str) -> None:
    plt.boxplot(Js, notch=True)
    ax = plt.gca()
    ax.hlines(base_J, 0.5, 1.5, color="red", linestyle="dotted")
    plt.ylabel("J value")
    plt.yscale("log")
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(out_filename)

def plot_Js(Js: dict, base_Js: dict, out_filename: str) -> None:
    fields_names = {
        "data/turing_winners/sbert-abstracts-ordered": "CS",
        "data/nobel_winners/chemistry/abstracts-ordered": "Chemistry",
        "data/nobel_winners/economics/abstracts-ordered": "Economics",
        "data/nobel_winners/medicine/abstracts-ordered": "Medicine",
        "data/nobel_winners/physics/abstracts-ordered": "Physics"
    }

    plt.boxplot(Js.values(), notch=True)
    ax = plt.gca()
    labels = []
    for i, field in enumerate(Js):
        ax.hlines(base_Js[field], i + 0.5, i + 1.5, color="red", linestyle="dotted")
        labels.append(fields_names[field])
    
    plt.xticks([i + 1 for i in range(len(Js))], labels)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_filename)


def plot_p_vals_hist(p_vals: List, out_filename: str) -> None:
    plt.hist(p_vals, bins=[i * 0.01 for i in range(10)])
    plt.savefig(out_filename)

if __name__ == "__main__":
    vecs_path_list = [
        "data/turing_winners/sbert-abstracts-ordered",
        "data/nobel_winners/chemistry/abstracts-ordered",
        "data/nobel_winners/economics/abstracts-ordered",
        "data/nobel_winners/medicine/abstracts-ordered",
        "data/nobel_winners/physics/abstracts-ordered"
    ]

    vecs_path_list_rand = [
        "data/nobel_winners/cs-random/abstracts-ordered",
        "data/nobel_winners/chemistry-random/abstracts-ordered",
        "data/nobel_winners/economics-random/abstracts-ordered",
        "data/nobel_winners/medicine-random/sbert-abstracts-ordered",
        "data/nobel_winners/physics-random/abstracts-ordered"
    ]
    #print(flatten_dict({"A": [[0, 1], [2, 3]], "B": [[3, 4], [5, 6]]}))
    #assert False
    #vecs_dict = get_vecs_field(vecs_path_list_rand, select_all=True)
    #with open("./data/sbert-vecs-dict-rand.p", "wb") as out_f: 
        #pickle.dump(vecs_dict, out_f)

    fields_dict, vecs_dict = pickle.load(open("./data/sbert-vecs-dict-rand.p", "rb"))
    #pdb.set_trace()
    #fields_dict, scientists_sample, vecs_dict = pickle.load(open("./data/sbert-vecs-dict-rand.p", "rb"))
    # #fake_dict = {"A": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], "B": [[100, 110], [120, 130], [140, 150]]}
    # #print(shuffle_fields(fake_dict))
    #Js, base_Js, p_vals = shuffle_samples_scientists_all(fields_dict)
    
    #assert False
    # plot_Js(Js, base_Js, "mult-field-Js-rand.png")
    # assert False
    
    #scientist_dict = sample_random_papers_by_scientist(fields_dict, "data/turing_winners/sbert-abstracts-ordered", select_all=True)
    p_val, Js, base_J = shuffle_Js(vecs_dict)
    #p_val, Js, base_J = shuffle_Js(scientist_dict)
    plot_Js_box_whisker(Js, base_J, "all-papers-J-rand.eps")
    plot_Js_box_whisker(Js, base_J, "all-papers-J-rand.png")

    #p_vals = shuffle_samples(fields_dict, num_samples=100, num_shuffles=10000)
    #print(p_vals)
    #plot_p_vals_hist(p_vals, "samples-hist.png")
