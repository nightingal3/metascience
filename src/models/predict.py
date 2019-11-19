import ast
import copy
import csv
from functools import reduce, lru_cache
import heapq
import random
from statistics import stdev
from typing import List, Callable, Generic

import numpy as np
from sklearn import preprocessing, neighbors, metrics
import statsmodels.stats.api as sms
import pdb


def get_attested_order(vecs_filename: str) -> List:
    rev_order = []
    with open(vecs_filename, "r") as vecs_file:
        reader = csv.reader(vecs_file)
        for row in reader:
            rev_order.append(ast.literal_eval(row[1]))
    return rev_order[::-1]


def rank_on_1NN(emerged: List, unemerged: List) -> List:
    closest = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sim = get_sim(emerged_vec, candidate_vec)
            if sim > closest[tuple(candidate_vec)]:
                closest[tuple(candidate_vec)] = sim[0][0]

    return [
        list(item[0])
        for item in sorted(
            [item for item in closest.items()], key=lambda x: x[1], reverse=True
        )
    ]


# Rank on local is implemented by limiting the emerged list to points that emerged at the last timestep
def rank_on_prototype(emerged: List, unemerged: List) -> List:
    closest = {tuple(vec): 0 for vec in unemerged}

    proto_vec = get_prototype(emerged)
    for candidate_vec in unemerged:
        sim = get_sim(candidate_vec, proto_vec)
        closest[tuple(candidate_vec)] = sim[0][0]
    return [
        list(item[0])
        for item in sorted(
            [item for item in closest.items()], key=lambda x: x[1], reverse=True
        )
    ]


def get_prototype(vecs: List) -> List:
    sum_vec = reduce(np.add, vecs)
    return np.divide(sum_vec, len(vecs))


def rank_on_progenitor(progenitor: tuple) -> Callable:
    def rank(emerged: List, unemerged: List) -> List:
        closest = {tuple(vec): 0 for vec in unemerged}
        for vec in unemerged:
            sim = get_sim(progenitor, vec)
            closest[tuple(vec)] = sim[0][0]

        return [
            list(item[0])
            for item in sorted(
                [item for item in closest.items()], key=lambda x: x[1], reverse=True
            )
        ]

    return rank


def rank_on_exemplar(emerged: List, unemerged: List) -> List:
    sum_sim = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sum_sim[tuple(candidate_vec)] += get_sim(emerged_vec, candidate_vec)

    return [
        list(item[0])
        for item in sorted(
            [item for item in sum_sim.items()], key=lambda x: x[1], reverse=True
        )
    ]


def get_sim(vec_1: List, vec_2: List) -> float:
    return metrics.pairwise.cosine_similarity(
        np.asarray(vec_1).reshape(1, -1), np.asarray(vec_2).reshape(1, -1)
    )


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


def get_random(emerged: List, unemerged: List) -> List:
    random.seed()
    random.shuffle(unemerged)
    return unemerged


def predict_seq(
    all_vecs: List,
    emergence_order: dict,
    ranking_func: Callable,
    error_func: Callable,
    ranking_model: Generic = None,
    ranking_type: str = "global"
) -> tuple:
    num_timesteps = max(emergence_order.keys())
    emerged_papers = []
    cumulative_err = 0
    rank_diff_per_timestep = []

    # Assume a single starting point in the space for no
    for t in range(num_timesteps):
        prev_papers = None
        curr_papers = emergence_order[t]
        next_papers = emergence_order[t + 1]
        emerged_papers.extend(curr_papers)

        last_papers = emerged_papers
        if (
            prev_papers and ranking_type == "local"
        ):  # only factor in the previous timestep for local algos
            last_papers = prev_papers

        available_papers = all_vecs[:]
        
        rank_err = 0

        for paper_vec in curr_papers:
            if ranking_type == "CRP":
                predicted_order = ranking_model.rank_on_clusters(last_papers, available_papers)
            else:
                predicted_order = ranking_func(last_papers, available_papers)
            prediction_error = error_func(predicted_order, next_papers)
            rank_err += prediction_error
           
            random_order = get_random(last_papers, available_papers)
            random_rank_err = error_func(random_order, next_papers)
            rank_diff_per_timestep.append(random_rank_err - prediction_error)

        rank_err = rank_err / len(
            curr_papers
        )  # Just take the average rank error at timestep?
        cumulative_err += rank_err

        prev_papers = curr_papers
        for paper in emerged_papers:
            available_papers.remove(
                list(paper)
            )  # Not using list comprehension so duplicates are preserved

        if ranking_type == "CRP":
            ranking_model.update_clusters(prev_papers, len(prev_papers))

    return cumulative_err / len(all_vecs), rank_diff_per_timestep


def get_rank_score_deprecated(
    predicted: np.ndarray, actual: np.ndarray, next_index: int
) -> int:
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


# Returns (p value, upper 95% confidence interval, lower confidence interval)
def shuffle_test(n_iter: int, target_val: int, emergence_order: dict, vecs_filename: str, order_filename: str, return_raw_counts: bool = False) -> tuple:
    higher = 0
    lower = 0
    cumulative_rank_diffs = []
    trial_timestep_rank_diff = None

    for i in range(n_iter):
        print(i)
        attested_order = get_attested_order(vecs_filename)
        emergence_order = get_emergence_order(order_filename)

        random.seed()
        rand_val, rank_diff_at_timestep = predict_seq(
            attested_order, emergence_order, get_random, get_rank_score_avg
        )
       
        cumulative_rank_diffs.append(rand_val - target_val)

        if trial_timestep_rank_diff == None:
            trial_timestep_rank_diff = rank_diff_at_timestep
        else:
            trial_timestep_rank_diff = [sum(x) for x in zip(trial_timestep_rank_diff, rank_diff_at_timestep)]

        if rand_val > target_val: # I reversed the p value by mistake. The p value returned is actually 1 - p
            higher += 1
        else:
            lower += 1

    avg_rank_diff = sum(cumulative_rank_diffs) / len(cumulative_rank_diffs)
    avg_rank_diff_timesteps = [i / n_iter for i in trial_timestep_rank_diff]

    upper_conf_interval, lower_conf_interval = sms.DescrStatsW(
        cumulative_rank_diffs
    ).tconfint_mean()
    if return_raw_counts:
        return [lower, higher]
    else:
        return (
            float(lower) / n_iter, # I reversed the p value by mistake. The p value returned is actually 1 - p
            avg_rank_diff,
            upper_conf_interval,
            lower_conf_interval,
            avg_rank_diff_timesteps
        )


if __name__ == "__main__":
    all_vecs = get_attested_order("data/hinton_paper_vectors.csv")
    emergence_order = get_emergence_order("data/ordered_by_date.csv")
    rank_on_prog = rank_on_progenitor(all_vecs[0])
    # all_vecs = get_attested_order("../../data/testing_rank.csv")
    # emergence_order = get_emergence_order("../../data/testing_rank.csv")

    """error, _ = predict_seq(
        all_vecs,
        emergence_order,
        rank_on_1NN,
        get_rank_score_avg,
        ranking_type="global",
    )"""
    
    error = 5
    print(shuffle_test(n_iter=100, target_val=error, emergence_order=emergence_order))
