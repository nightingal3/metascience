import ast
import copy
import csv
from functools import reduce, lru_cache
import heapq
import random
from statistics import stdev
from typing import List, Callable, Generic

import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing, metrics
import statsmodels.stats.api as sms
import pdb


def get_attested_order(vecs_filename: str, vecs_col: int = 1) -> List:
    rev_order = []
    with open(vecs_filename, "r") as vecs_file:
        reader = csv.reader(vecs_file)
        for row in reader:
            rev_order.append(ast.literal_eval(row[vecs_col]))
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

def make_rank_on_knn(k: int) -> Callable:
    curr_iter = 0
    def rank_on_knn(emerged: List, unemerged: List) -> List:
        nonlocal curr_iter
        curr_iter += 1
        closest = {tuple(vec): 0 for vec in unemerged}
        curr_k = min(k, curr_iter)
        for candidate_vec in unemerged:
            total_dist = 0
            pdb.set_trace()
            closest_vecs = emerged[distance.cosine(np.asarray(emerged), candidate_vec).argmin(curr_k)]
            for vec in closest_vecs:
                total_dist += distance.cosine(candidate_vec, vec)
            closest[tuple(candidate_vec)] = total_dist

        return [
            list(item[0])
            for item in sorted(
                [item for item in closest.items()], key=lambda x: x[1], reverse=True
            )
        ]

    return rank_on_knn


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


def get_emergence_order(emergence_filename: str, vecs_col: int = 1) -> dict:
    emergence_order = {}
    with open(emergence_filename, "r") as order_file:
        reader = csv.reader(order_file)
        for row in reader:
            if int(row[0]) not in emergence_order:
                emergence_order[int(row[0])] = [tuple(ast.literal_eval(row[vecs_col]))]
            else:
                emergence_order[int(row[0])].append(tuple(ast.literal_eval(row[vecs_col])))

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
        for paper in emerged_papers:
            available_papers.remove(
                list(paper)
            )  # Not using list comprehension so duplicates are preserved

        rank_err = 0

        for paper_vec in curr_papers:
            if ranking_type == "CRP":
                predicted_order = ranking_model.rank_on_clusters(last_papers, available_papers)
            else:
                predicted_order = ranking_func(last_papers, available_papers)
            
            #pdb.set_trace()
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
        
        if ranking_type == "CRP":
            ranking_model.update_clusters(prev_papers, len(prev_papers))

    return cumulative_err / len(all_vecs), rank_diff_per_timestep



def predict_seq_multi(
    all_vecs: List,
    emergence_order: dict,
    ranking_func_list: List,
    error_func: Callable,
    ranking_types: List,
    ranking_model: Generic = None
) -> tuple:
    num_timesteps = max(emergence_order.keys())
    emerged_papers = []
    cumulative_err = {ranking_func: 0 for ranking_func in ranking_func_list}
    rank_diff_per_timestep = {ranking_func: [] for ranking_func in ranking_func_list}

    # Assume a single starting point in the space for now
    for ranking_func, ranking_type in zip(ranking_func_list, ranking_types):
        emerged_papers = []
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
                rank_diff_per_timestep[ranking_func].append(random_rank_err - prediction_error)

            rank_err = rank_err / len(
                curr_papers
            )  # Just take the average rank error at timestep?
            cumulative_err[ranking_func] += rank_err

            prev_papers = curr_papers
            for paper in emerged_papers:
                available_papers.remove(
                    list(paper)
                )  # Not using list comprehension so duplicates are preserved

            if ranking_type == "CRP":
                ranking_model.update_clusters(prev_papers, len(prev_papers))

    return cumulative_err / len(all_vecs), rank_diff_per_timestep

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
    _attested_order = get_attested_order(vecs_filename)
    _emergence_order = get_emergence_order(order_filename)

    for i in range(n_iter):
        attested_order = _attested_order
        emergence_order = _emergence_order

        random.seed()
        rand_val, rank_diff_at_timestep = predict_seq(
            attested_order, emergence_order, get_random, get_rank_score_avg
        )
       
        cumulative_rank_diffs.append(rand_val - target_val)

        if trial_timestep_rank_diff == None:
            trial_timestep_rank_diff = rank_diff_at_timestep
        else:
            trial_timestep_rank_diff = [sum(x) for x in zip(trial_timestep_rank_diff, rank_diff_at_timestep)]

        print("random error: ", rand_val)
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
    vecs_filename = "data/turing_winners/vecs-abstracts-only/Judea-Pearl.csv"
    order_filename = "data/turing_winners/vecs-abstracts-ordered/Judea-Pearl.csv"

    #vecs_filename = "data/turing_winners/abstracts/geoff/Geoff-Hinton-title-vecs.csv"
    #order_filename = "data/turing_winners/abstracts/geoff/Geoff-Hinton-title-ordered.csv"
    all_vecs = get_attested_order(vecs_filename)
    emergence_order = get_emergence_order(order_filename)
    rank_on_prog = rank_on_progenitor(all_vecs[0])
    # all_vecs = get_attested_order("../../data/testing_rank.csv")
    # emergence_order = get_emergence_order("../../data/testing_rank.csv")

    error, rank_diff_per_timestep = predict_seq(
        all_vecs,
        emergence_order,
        rank_on_prog,
        get_rank_score_best,
        ranking_type="global",
    )
    print("error: ", error)
    print("rank diff per timestep: ", rank_diff_per_timestep)

    print(shuffle_test(n_iter=100, target_val=error, emergence_order=emergence_order, vecs_filename=vecs_filename, order_filename=order_filename))
