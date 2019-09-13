import ast
import copy
import csv
from functools import reduce
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

    return [vec for cos_sim, vec in heapq.nlargest(len(max_heap), max_heap)]

def get_kNN(k: int, all_vecs: List, vec: List) -> List:
    pass

def get_emergence_order(emergence_filename: str) -> dict:
    emergence_order = {}
    with open(emergence_filename, "r") as order_file:
        reader = csv.reader(order_file)
        for row in reader: 
            if int(row[0]) not in emergence_order:
                emergence_order[int(row[0])] = [tuple(ast.literal_eval(row[1]))]
            else:
                emergence_order[int(row[0])].append(tuple(ast.literal_eval(row[1])))

    return emergence_order

def get_random(vecs_list: List, vec: List) -> List:
    random.shuffle(vecs_list)
    return vecs_list

def predict_seq(all_vecs: List, emergence_order: dict, ranking_func: Callable) -> int:
    curr_ind = -1
    cumulative_error = 0
    eligible_vecs = all_vecs
    eligible_vecs.remove(all_vecs[-1])
    previously_predicted = [all_vecs[-1]]

    while len(eligible_vecs) > 0 and curr_ind > -len(all_vecs) + 1:
        # Just make one prediction for each vector in attested order to not propagate error
        next_index = -curr_ind 
        curr_vec = all_vecs[curr_ind]
        predicted = ranking_func(eligible_vecs, curr_vec)
        assert len(predicted) == len(eligible_vecs)
        actual = emergence_order[next_index]

        print("next timestep: ", next_index)
        rank = get_rank_score_best(predicted, actual, previously_predicted)
        print(rank)
        eligible_vecs.remove(predicted[0])
        previously_predicted.append(predicted[0])
        cumulative_error += rank
        curr_ind -= 1

    return cumulative_error

def _predict_seq(all_vecs: List, emergence_order: dict, ranking_func: Callable, error_func: Callable) -> int:
    num_timesteps = max(emergence_order.keys())
    available_vecs = all_vecs 
    cumulative_err = 0
    removed = []

    # Assume a single starting point in the space for now
    for t in range(num_timesteps):
        curr_papers = emergence_order[t]
        next_papers = emergence_order[t + 1]
        rank_err = 0 
        
        for paper_vec in curr_papers:
            available_vecs = available_vecs[:]  # Don't include self in possible predictions
            available_vecs.remove(list(paper_vec)) # ...but keep duplicates
            
            predicted_order = ranking_func(available_vecs, paper_vec)
            rank_err += error_func(predicted_order, next_papers)

        available_vecs = [vec for vec in available_vecs if vec not in curr_papers]
        removed.extend(curr_papers)

        rank_err = rank_err / len(curr_papers) # Just take the average rank error at timestep?
        cumulative_err += rank_err
    
    return cumulative_err / len(all_vecs)

def get_rank_score_deprecated(predicted: np.ndarray, actual: np.ndarray, next_index: int) -> int:
    return predicted.index(actual)

def get_rank_score_avg(predicted: np.ndarray, actual: List) -> int:
    rank_diff = 0
    for vec in actual:
        rank_diff += predicted.index(list(vec))

    return rank_diff / len(actual)

def get_rank_score_best(predicted: np.ndarray, actual: List) -> int:
    ranks = [predicted.index(list(vec)) for vec in actual]
    return min(ranks)

def get_rank_score_worst(predicted: np.ndarray, actual: List) -> int:
    ranks = [predicted.index(list(vec)) for vec in actual]
    return max(ranks)

def shuffle_test(n_iter: int, target_val: int, emergence_order: dict) -> float:
    higher = 0
    lower = 0

    for i in range(n_iter): 
        print(i)
        attested_order = get_attested_order("../../data/hinton_paper_vectors.csv")
        emergence_order = get_emergence_order("../../data/ordered_by_date.csv")

        rand_val = _predict_seq(attested_order, emergence_order, get_random, get_rank_score_worst)
        if rand_val > target_val:
            higher += 1
        else:
            lower += 1

    return float(lower) / n_iter 

if __name__ == "__main__":
    all_vecs = get_attested_order("../../data/hinton_paper_vectors.csv")
    emergence_order = get_emergence_order("../../data/ordered_by_date.csv")

    error = _predict_seq(all_vecs, emergence_order, get_1NN, get_rank_score_worst)
    print("error 1NN: ", error)
    error1 = _predict_seq(all_vecs, emergence_order, get_random, get_rank_score_worst)
    print("error random: ", error1)
    print(shuffle_test(n_iter=10000, target_val=error, emergence_order=emergence_order))
