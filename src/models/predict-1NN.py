import ast
import copy
import csv
import heapq
import random
from typing import List, Callable

import numpy as np
from sklearn import preprocessing, neighbors, metrics

def get_attested_order(vecs_filename: str) -> List:
    rev_order = []
    with open(vecs_filename, "r") as vecs_file:
        reader = csv.reader(vecs_file)
        for row in reader:
            rev_order.append(ast.literal_eval(row[1]))
    return rev_order[::-1]

def get_1NN(all_vecs: List, vec: List) -> List:
    max_heap = []

    for curr_vec in all_vecs:
        cos_sim = metrics.pairwise.cosine_similarity(np.asarray(curr_vec).reshape(1, -1), np.asarray(vec).reshape(1, -1))
        heapq.heappush(max_heap, (cos_sim, curr_vec))

    return [vec for cos_sim, vec in heapq.nlargest(len(max_heap), max_heap)][1:]

def get_random(vecs_list: List, vec: List) -> List:
    random.shuffle(vecs_list)
    return vecs_list

def predict_seq(all_vecs: List, ranking_func: Callable) -> int:
    curr_ind = -1
    cumulative_error = 0
    eligible_vecs = all_vecs

    while len(eligible_vecs) > 0 and curr_ind > -len(all_vecs) + 1:
        # Just make one prediction for each vector in attested order to not propagate error
        curr_vec = all_vecs[curr_ind]
        next_vec = all_vecs[curr_ind - 1]
        predicted = ranking_func(eligible_vecs, curr_vec)
        rank = get_rank_score(predicted, next_vec)
        eligible_vecs.remove(predicted[0])
        cumulative_error += rank
        curr_ind -= 1

    return cumulative_error

def get_rank_score(predicted: np.ndarray, actual: np.ndarray) -> int:
    return predicted.index(actual)

def shuffle_test(n_iter: int, target_val: int) -> float:
    higher = 0
    lower = 0

    for i in range(n_iter): 
        attested_order = get_attested_order("../../data/hinton_paper_vectors.csv")
        rand_val =  predict_seq(attested_order, get_random)
        if rand_val > target_val:
            higher += 1
        else:
            lower += 1

    return float(lower) / n_iter 

if __name__ == "__main__":
    attested_order = get_attested_order("../../data/hinton_paper_vectors.csv")
    error = predict_seq(attested_order, get_1NN)
    print(shuffle_test(n_iter=10000, target_val=error))
