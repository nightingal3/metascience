import argparse
import ast
import copy
import csv
from functools import reduce, lru_cache
import heapq
import random
from statistics import stdev
from typing import List, Callable, Generic

import math
import numpy as np
from scipy.spatial import distance
from scipy.special import softmax
from scipy.stats import ttest_1samp
from sklearn import preprocessing, metrics
import statsmodels.stats.api as sms
import pdb
import os
import pickle
from pprint import pprint

def get_attested_order(vecs_filename: str, vecs_col: int = 1, label: bool = False, multicols: bool = False) -> List:
    rev_order = []
    if ".csv" in vecs_filename:
        with open(vecs_filename, "r") as vecs_file:
            reader = csv.reader(vecs_file)
            for row in reader:
                if multicols: 
                    try:
                        vec_lst = ast.literal_eval(", ".join(row[vecs_col:]))
                        rev_order.append(vec_lst)
                    except:
                        pdb.set_trace()
                        print(row[vecs_col:])
                        assert False
                else:
                    if not label:
                        rev_order.append(ast.literal_eval(row[vecs_col]))
                    else:
                        rev_order.append(row[vecs_col])
        return rev_order[::-1]        

def rank_on_1NN(emerged: List, unemerged: List, keep_sim: bool = False, dist="nbhd") -> List:
    closest = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sim = get_sim(emerged_vec, candidate_vec, dist=dist)
            if sim > closest[tuple(candidate_vec)]:
                closest[tuple(candidate_vec)] = sim[0][0]

    if keep_sim:
        return [
            list(item)
            for item in sorted(
                [item for item in closest.items()], key=lambda x: x[1], reverse=True
            )
        ]
    else:
        return [
            list(item[0])
            for item in sorted(
                [item for item in closest.items()], key=lambda x: x[1], reverse=True
            )
        ]

def make_rank_on_knn(k: int) -> Callable:
    curr_iter = 0
    emerged_num = 0
    def rank_on_knn(emerged: List, unemerged: List, keep_sim: bool = False, dist="nbhd") -> List:
        nonlocal curr_iter
        nonlocal emerged_num

        if len(emerged) != emerged_num: # A timestep has gone by
            emerged_num = len(emerged)
            curr_iter += 1

        closest = {tuple(vec): 0 for vec in unemerged}
        curr_k = min(k, curr_iter)

        for candidate_vec in unemerged:
            sim_sum = 0
            if dist == "nbhd":
                #dist_func = lambda u, v: -np.exp(-(distance.euclidean(u, v) ** 2))
                dist_func = lambda u, v: -np.exp(-(distance.euclidean(u, v)))
                closest_vecs = np.argpartition(distance.cdist(np.array([candidate_vec]), np.asarray(emerged), metric=dist_func), curr_k - 1)[0][:curr_k]
            else:
                closest_vecs = np.argpartition(distance.cdist(np.array([candidate_vec]), np.asarray(emerged), metric=dist), curr_k - 1)[0][:curr_k]

            #print("closest: ", closest_vecs)
            for vec_ind in closest_vecs:
                #print("added to sim sum: ", get_sim(candidate_vec, emerged[vec_ind])[0][0])
                sim_sum += get_sim(candidate_vec, emerged[vec_ind], dist=dist)[0][0]
            closest[tuple(candidate_vec)] = sim_sum
            #print("sum: ", sim_sum)
        #print(max(closest.values()))
    
        if keep_sim:
            return [
                item
                for item in sorted(
                    [item for item in closest.items()], key=lambda x: x[1], reverse=True
                )
            ]
        else:
            return [
                list(item[0])
                for item in sorted(
                    [item for item in closest.items()], key=lambda x: x[1], reverse=True
                )
            ]

    return rank_on_knn


# Rank on local is implemented by limiting the emerged list to points that emerged at the last timestep
def rank_on_prototype(emerged: List, unemerged: List, keep_sim: bool = False, dist="nbhd") -> List:
    closest = {tuple(vec): 0 for vec in unemerged}

    proto_vec = get_prototype(emerged)
    for candidate_vec in unemerged:
        sim = get_sim(candidate_vec, proto_vec, dist=dist)
        closest[tuple(candidate_vec)] = sim[0][0]

    if keep_sim: 
        return [
            list(item)
            for item in sorted(
                [item for item in closest.items()], key=lambda x: x[1], reverse=True
            )
        ]
    return [
        list(item[0])
        for item in sorted(
            [item for item in closest.items()], key=lambda x: x[1], reverse=True
        )
    ]


def get_prototype(vecs: List) -> List:
    sum_vec = reduce(np.add, vecs)
    return np.divide(sum_vec, len(vecs))


def rank_on_progenitor(progenitor_list: List) -> Callable:
    def rank(emerged: List, unemerged: List, keep_sim: bool = False, dist="nbhd") -> List:
        closest = {tuple(vec): 0 for vec in unemerged}
        for vec in unemerged:
            for progenitor in progenitor_list:
                sim = get_sim(progenitor, vec, dist=dist)
                closest[tuple(vec)] = sim[0][0]

        if keep_sim:
            return [
                list(item)
                for item in sorted(
                    [item for item in closest.items()], key=lambda x: x[1], reverse=True
                )
            ]
        else:
            return [
                list(item[0])
                for item in sorted(
                    [item for item in closest.items()], key=lambda x: x[1], reverse=True
                )
            ]
    return rank

def rank_on_exemplar(emerged: List, unemerged: List, keep_sim: bool = False, dist="nbhd", s=1.0) -> List:
    sum_sim = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sum_sim[tuple(candidate_vec)] += get_sim(emerged_vec, candidate_vec, dist=dist, s=s)[0][0]
    
    sum_sim = {k: v / len(emerged) for k, v in sum_sim.items()}

    if keep_sim:
        return [
            list(item)
            for item in sorted(
                [item for item in sum_sim.items()], key=lambda x: x[1], reverse=True
            )
        ]
    else:
        return [
            list(item[0])
            for item in sorted(
                [item for item in sum_sim.items()], key=lambda x: x[1], reverse=True
            )
        ]

def make_rank_on_exemplar(s: float = 1.0) -> Callable:
    def rank_on_exemplar(emerged: List, unemerged: List, keep_sim: bool = False, dist="nbhd", s=s) -> List:
        sum_sim = {tuple(vec): 0 for vec in unemerged}

        for emerged_vec in emerged:
            for candidate_vec in unemerged:
                sum_sim[tuple(candidate_vec)] += get_sim(emerged_vec, candidate_vec, dist=dist, s=s)[0][0]
        
        sum_sim = {k: v / len(emerged) for k, v in sum_sim.items()}

        if keep_sim:
            return [
                list(item)
                for item in sorted(
                    [item for item in sum_sim.items()], key=lambda x: x[1], reverse=True
                )
            ]
        else:
            return [
                list(item[0])
                for item in sorted(
                    [item for item in sum_sim.items()], key=lambda x: x[1], reverse=True
                )
            ]
    

    return rank_on_exemplar


def get_sim(vec_1: List, vec_2: List, dist="nbhd", s=1.0) -> float:
    vec_1 = np.asarray(vec_1).reshape(1, -1)
    vec_2 = np.asarray(vec_2).reshape(1, -1)
    if dist == "cosine":
        #if metrics.pairwise.cosine_similarity(np.asarray(vec_1).reshape(1, -1), np.asarray(vec_2).reshape(1, -1)) > 10:
            #pdb.set_trace()
        return metrics.pairwise.cosine_similarity(
            vec_1, vec_2
        )
    elif dist == "euclidean":
        return 1/(1 + metrics.pairwise.euclidean_distances(
            vec_1, vec_2
        ))
    elif dist == "nbhd":
        euc_dist = metrics.pairwise.euclidean_distances(vec_1, vec_2)
        #return np.exp(s * -(euc_dist ** 2))
        return np.exp(s * -(euc_dist)) # underflow issues - using regular Euclidean distance (same ranking as squared)


def get_emergence_order(emergence_filename: str, vecs_col: int = 1, multicols: bool = False) -> dict:
    emergence_order = {}
    seen = set()
    with open(emergence_filename, "r") as order_file:
        reader = csv.reader(order_file)
        for row in reader:
            if multicols:
                if int(float(row[0])) not in emergence_order:
                    vec = tuple(ast.literal_eval(", ".join(row[vecs_col:])))
                    emergence_order[int(float(row[0]))] = [tuple(ast.literal_eval(", ".join(row[vecs_col:])))]
                else:
                    emergence_order[int(float(row[0]))].append(tuple(ast.literal_eval(", ".join(row[vecs_col:]))))
            else:
                if int(row[0]) not in emergence_order:
                    emergence_order[int(float(row[0]))] = [tuple(ast.literal_eval(row[vecs_col]))]
                else:
                    emergence_order[int(float(row[0]))].append(tuple(ast.literal_eval(row[vecs_col])))

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
    ranking_type: str = "global",
    expected_wins: bool = False,
    memoized_pred_ranks: List = []
) -> tuple:
    num_timesteps = max(emergence_order.keys())
    emerged_papers = []
    cumulative_err = 0
    rank_diff_per_timestep = []
    pred_ranks_vec = []
    rand_ranks_vec = []

    # Assume a single starting point in the space for now
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
            )  
        available_papers = [tuple(x) for x in available_papers]
        available_papers = list(set(available_papers))
        available_papers = [list(x) for x in available_papers]
        
        if len(available_papers) == 0:
            if expected_wins:
                return expected_model_better_rand, pred_ranks_vec
    
            return cumulative_err / num_timesteps, rank_diff_per_timestep, expected_model_better_rand


        rank_err = 0

        if ranking_type == "CRP":
            predicted_order = ranking_model.rank_on_clusters(last_papers, next_papers)
        else:
            if memoized_pred_ranks == []:
                predicted_order = ranking_func(last_papers, available_papers)
                if len(predicted_order) < len(available_papers):
                    pdb.set_trace()
                    predicted_order = ranking_func(last_papers, available_papers)

        
        #print("next papers: ", len(next_papers))
        if memoized_pred_ranks == []:
            prediction_error = error_func(predicted_order, next_papers)
            rank_err += prediction_error

        random_order = get_random(last_papers, available_papers)
        random_rank_err = error_func(random_order, next_papers)
        #print("random error: ", random_rank_err)
        if memoized_pred_ranks == []:
            rank_diff_per_timestep.append(random_rank_err - prediction_error)

            #print("RANK DIFF: ", random_rank_err - prediction_error)
            cumulative_err += random_rank_err - prediction_error

        for paper in next_papers:
            other_papers = [p for p in next_papers if p != paper]
            if memoized_pred_ranks == []:
                pred_others_removed = predicted_order[:]
            rand_others_removed = random_order[:]

            for p in other_papers:
                if memoized_pred_ranks == []:
                    try:
                        pred_others_removed.remove(list(p))
                    except:
                        continue
                try:
                    rand_others_removed.remove(list(p))
                except:
                    continue
            if memoized_pred_ranks == []:
                pred_ranks_vec.append(pred_others_removed.index(list(paper)))
            rand_ranks_vec.append(rand_others_removed.index(list(paper)))

        prev_papers = curr_papers
        
        if ranking_type == "CRP":
            ranking_model.update_clusters(prev_papers, len(prev_papers))
    if memoized_pred_ranks != []:
        if len(rand_ranks_vec) == 0:
            expected_model_better_rand = 0.5
        else:
            expected_model_better_rand = sum(indicator(rand_ranks_vec, memoized_pred_ranks))/len(rand_ranks_vec)
    else:
        if len(rand_ranks_vec) == 0:
            expected_model_better_rand = 0.5
        else:
            expected_model_better_rand = sum(indicator(rand_ranks_vec, pred_ranks_vec))/len(rand_ranks_vec)
        
    #print("CUMULATIVE ERR:", cumulative_err / num_timesteps)
    #print("PRED RANKS: ", pred_ranks_vec[:5])
    #print("RAND RANKS: ", rand_ranks_vec[:5])

    if expected_wins:
        return expected_model_better_rand, pred_ranks_vec
    
    return cumulative_err / num_timesteps, rank_diff_per_timestep, expected_model_better_rand


def predict_seq_multi(
    all_vecs: List,
    emergence_order: dict,
    ranking_func_dict: dict,
    error_func: Callable,
    ranking_types: List,
    ranking_model: Generic = None
) -> tuple:
    num_timesteps = max(emergence_order.keys())
    emerged_papers = []
    cumulative_err = {ranking_func: 0 for ranking_func in ranking_func_list}
    rank_diff_per_timestep = {ranking_func: [] for ranking_func in ranking_func_list}
    expected_models_better_rand = {}

    # Assume a single starting point in the space for now
    for (model_name, ranking_func), ranking_type in zip(ranking_func_dict.items(), ranking_types):
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
        

    return cumulative_err / num_timesteps, rank_diff_per_timestep

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


def indicator(rand_ranks: List, model_ranks: List) -> List:
    return [1 if rand_rank > model_rank else 0 for rand_rank, model_rank in zip(rand_ranks, model_ranks)]


def get_probability_score_multi(emergence_order: dict, all_vecs: List, ranking_funcs: dict, ranking_types: List, carry_error: bool = False, suppress_print: bool = True) -> tuple:
    last_timestep = max(emergence_order.keys())
    log_Ls = [0] * len(ranking_funcs)
    order = []
    emerged_papers = [[] for i in range(len(ranking_funcs))]

    if not suppress_print:
        print("TOTAL PAPERS: ", len(all_vecs))
    for t in range(last_timestep):
        if not suppress_print:
            print("TIMESTEP: ", t)
        for i, name in enumerate(ranking_funcs):
            curr_papers = emergence_order[t]

            #print("curr papers: ", len(curr_papers))
            next_papers = emergence_order[t + 1]
            if not carry_error:
                emerged_papers[i].extend(curr_papers)
                last_papers = emerged_papers[i]
                if ranking_types[i] == "local":
                    last_papers = curr_papers
            else:
                if t == 0:
                    emerged_papers[i].extend(emergence_order[0])
                last_papers = emerged_papers[i]
            
            available_papers = all_vecs[:]
            for paper in emerged_papers[i]:
                available_papers.remove(
                    list(paper)
                )  # Not using list comprehension so duplicates are preserved

            pred_and_sim = ranking_funcs[name](last_papers, available_papers, keep_sim=True)
            pred, sim = zip(*pred_and_sim)
            log_L = 0

            #sim_softmax = [prob / sum(sim) for prob in sim]
            sim_softmax = softmax(sim)

            if not carry_error:
                for vec in next_papers:
                    #pdb.set_trace()
                    next_indices = [pred.index(v) for v in next_papers]
                    found_index = pred.index(vec)

                    next_indices_excluded_self = [i for i in next_indices if i != found_index]
                    sim_others_excluded = [prob if i not in next_indices_excluded_self else 0 for i, prob in enumerate(sim_softmax)]
                    sim_others_excluded = [prob/sum(sim_others_excluded) for prob in sim_others_excluded]
                    #print(f"{found_index}: {sim_others_excluded[found_index]}, len: {len([i for i in sim_others_excluded if i != 0])}") 
                    #if sim_others_excluded[found_index] == 0:
                        #pdb.set_trace()
                    log_L += np.log(sim_others_excluded[found_index])
                    
            
            if carry_error:
                for vec in pred[:len(next_papers)]:
                    closest_ind = np.argpartition(distance.cdist(np.array([vec]), np.asarray(emerged_papers[i]), metric="cosine"), 0)[0][0]
                    emerged_papers[i].append(vec)
                
            log_Ls[i] += log_L
            order.append(name)

    return log_Ls, order

def get_probability_score(emergence_order: dict, all_vecs: dict, ranking_func: Callable, ranking_type: str = "global", carry_error: bool = False, labels: List = [], return_log_L_only: bool = False, suppress_print: bool = True) -> tuple:
    last_timestep = max(emergence_order.keys())
    log_L = 0
    emerged_papers = []
    tails = []
    if not suppress_print:
        print("TOTAL PAPERS: ", len(all_vecs))
    for t in range(last_timestep):
        #print("TIMESTEP: ", t)
        curr_papers = emergence_order[t]

        #print("curr papers: ", len(curr_papers))
        next_papers = emergence_order[t + 1]
        if not carry_error:
            emerged_papers.extend(curr_papers)
            last_papers = emerged_papers
            if ranking_type == "local":
                last_papers = curr_papers
        else:
            if t == 0:
                emerged_papers.extend(emergence_order[0])
            last_papers = emerged_papers
        
        available_papers = all_vecs[:]
        for paper in emerged_papers:
            available_papers.remove(
                list(paper)
            )  # Not using list comprehension so duplicates are preserved

        pred_and_sim = ranking_func(last_papers, available_papers, keep_sim=True)
        pred, sim = zip(*pred_and_sim)

        #sim_softmax = [prob / sum(sim) for prob in sim]
        sim_softmax = softmax(sim)

        if not carry_error:
            for vec in next_papers:
                #pdb.set_trace()
                next_indices = [pred.index(v) for v in next_papers]
                found_index = pred.index(vec)

                next_indices = [i for i in next_indices if i != found_index]
                sim_others_excluded = [prob if i not in next_indices else 0 for i, prob in enumerate(sim_softmax)]
                sim_others_excluded = [prob/sum(sim_others_excluded) for prob in sim_others_excluded]
                #print(f"{found_index}: {sim_others_excluded[found_index]}, len: {len([i for i in sim_others_excluded if i != 0])}") 
                #if sim_others_excluded[found_index] == 0:
                    #pdb.set_trace()
                log_L += np.log(sim_others_excluded[found_index])
                
        
        if carry_error:
            for vec in pred[:len(next_papers)]:
                closest_ind = np.argpartition(distance.cdist(np.array([vec]), np.asarray(emerged_papers), metric="cosine"), 0)[0][0]
                tails.append(closest_ind)
                emerged_papers.append(vec)

    if return_log_L_only:
        return log_L

    return log_L, emerged_papers, tails

def get_probability_rand(emergence_order: dict) -> float:
    """ Returns a fixed probability for a predicted sequence based on the number
    papers emerging at each timestep. 
    timesteps: [run_0, run_1, run_2...]
    """
    timesteps = make_timesteps(emergence_order)
    log_L = 0
    denom = sum(timesteps[1:])
    for timestep in timesteps[1:]:
        for i in range(timestep):
            log_L += np.log(1 / (denom - timestep + timesteps[0]))
            #print("val: ", 1/(denom - timestep + 1) , " denom: ", denom - timestep + 1)
        denom -= timestep
    return log_L

def make_timesteps(emergence_order: dict) -> float:
    runs = []
    for i in range(max(emergence_order.keys()) + 1):
        runs.append(len(emergence_order[i]))
    return runs

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

        if rand_val > target_val: 
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
            float(lower) / n_iter, 
            avg_rank_diff,
            upper_conf_interval,
            lower_conf_interval,
            avg_rank_diff_timesteps
        )


def shuffle_test_multi(n_iter: int, target_val_list: List, emergence_order: dict, vecs_filename: str, order_filename: str, return_raw_counts: bool = False) -> tuple:
    cumulative_rank_diffs = []
    trial_timestep_rank_diff = None
    _attested_order = get_attested_order(vecs_filename)
    _emergence_order = get_emergence_order(order_filename)
    pvals = []
    pvals_ratio = []
    avg_rank_diffs = []
    avg_rank_diff_timesteps_multi = []
    upper_conf_intervals = []
    upper_conf_intervals_ratio = []
    lower_conf_intervals = []
    lower_conf_intervals_ratio = []
    expected_wins = []

    for target_val in target_val_list:
        higher = 0
        lower = 0
        for i in range(n_iter):
            print(i)
            attested_order = _attested_order
            emergence_order = _emergence_order

            random.seed()
            rand_val, rank_diff_at_timestep, win_ratio = predict_seq(
                attested_order, emergence_order, get_random, get_rank_score_avg
            )
        
            cumulative_rank_diffs.append(rand_val - target_val)
            expected_wins.append(win_ratio)

            if trial_timestep_rank_diff == None:
                trial_timestep_rank_diff = rank_diff_at_timestep
            else:
                trial_timestep_rank_diff = [sum(x) for x in zip(trial_timestep_rank_diff, rank_diff_at_timestep)]

            if rand_val > target_val: 
                higher += 1 
            else:
                lower += 1
        

        avg_rank_diff = sum(cumulative_rank_diffs) / len(cumulative_rank_diffs)
        avg_rank_diffs.append(avg_rank_diff)
        if avg_rank_diff >= 0:
            pvals.append(float(lower) / n_iter)
        else:
            pvals.append(float(higher) / n_iter)
        avg_rank_diff_timesteps = [i / n_iter for i in trial_timestep_rank_diff]
        avg_rank_diff_timesteps_multi.append(avg_rank_diff_timesteps)
        upper_conf_interval, lower_conf_interval = sms.DescrStatsW(
            cumulative_rank_diffs
        ).tconfint_mean()
        p_val_wins = ttest_1samp(expected_wins, 0.5)
        pvals_ratio.append(p_val_wins)
        u_c_interval, l_c_interval = sms.DescrStatsW(expected_wins).tconfint_mean()

        upper_conf_intervals.append(upper_conf_interval)
        lower_conf_intervals.append(lower_conf_interval)
        upper_conf_intervals_ratio.append(u_c_interval)
        lower_conf_intervals_ratio.append(l_c_interval)

    if return_raw_counts:
        return [lower, higher]
    else:
        return (
            pvals,
            avg_rank_diffs,
            upper_conf_intervals,
            lower_conf_intervals,
            avg_rank_diff_timesteps_multi
        )

def shuffle_test_expected_vals(n_iter: int, ranking_funcs: dict, model_order: dict, vecs_filename: str, order_filename: str) -> tuple:
    _attested_order = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
    _emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)
    
    memoized_preds = {}
    win_ratios = {}

    for func_name in ranking_funcs:
        model_type = "local" if func_name == "Local" else "global"
        expected_ratio, pred_ranks_vec = predict_seq(
            _attested_order, _emergence_order, ranking_funcs[func_name], get_rank_score_avg, ranking_type=model_type, expected_wins=True
        )
        memoized_preds[func_name] = pred_ranks_vec
        win_ratios[func_name] = []

    for i in range(n_iter):
        print(i)
        attested_order = _attested_order
        emergence_order = _emergence_order

        random.seed()
        for func_name in memoized_preds:
            win_ratio, _ = predict_seq(
                attested_order, emergence_order, ranking_funcs[func_name], get_rank_score_avg, expected_wins=True, memoized_pred_ranks=memoized_preds[func_name]
            )
            win_ratios[func_name].append(win_ratio)

    pvals = []
    win_ratios_per_model = []
    yerrs = []
    tvals = []

    for i in range(max(model_order.keys()) + 1):
        func_name = model_order[i]
        avg_win_ratio = sum(win_ratios[func_name])/len(win_ratios[func_name])
        t_val_wins, p_val_wins = ttest_1samp(win_ratios[func_name], 0.5)
        upper_conf_interval, lower_conf_interval = sms.DescrStatsW(win_ratios[func_name]).tconfint_mean()
        pvals.append(p_val_wins)
        win_ratios_per_model.append(avg_win_ratio)
        yerrs.append(upper_conf_interval - lower_conf_interval)
        tvals.append(t_val_wins)
        print(func_name)
        print(avg_win_ratio)
        print(p_val_wins)

    return pvals, win_ratios_per_model, yerrs, tvals
        

def shuffle_emergence_years(all_vecs: List, emergence_order: dict) -> tuple:
    num_per_timestep = {key: len(emergence_order[key]) for key in emergence_order}
    new_emergence_order = {key: [] for key in emergence_order}
    shuffled_order = random.sample(all_vecs, len(all_vecs))
    i = 0

    for t in num_per_timestep:
        num_emerged = num_per_timestep[t]
        for _ in range(num_emerged):
            new_emergence_order[t].append(tuple(shuffled_order[i]))
            i += 1

    return shuffled_order, new_emergence_order

def evaluate_on_files(measure_type: str, test_only: bool, shuffle_emergence_order: bool, all_filenames: List, vecs_path: str, order_path: str, train_path: str, individual_path: str, out_path: str, s_vals_path: str, field: str, s_val: float, train_inds: List):
    models = {
        "1NN": {},
        "2NN": {},
        "3NN": {},
        "4NN": {},
        "5NN": {},
        "Prototype": {},
        "Progenitor": {},
        "Exemplar": {},
        "Exemplar (s=1)": {},
        "Exemplar (s=0.001)": {},
        "Exemplar (s=0.1)": {},
        "Local": {},
        "Null": {}
    }
    for i, filename in enumerate(all_filenames): 
        if filename.endswith(".csv"):
            print(filename)
            if test_only and i in train_inds:
                continue
            vecs_filename = os.path.join(vecs_path, filename)
            order_filename = os.path.join(order_path, filename)
            all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
            emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)
            if shuffle_emergence_order:
                all_vecs, emergence_order = shuffle_emergence_years(all_vecs, emergence_order)

            name_to_model = {
                "1NN": rank_on_1NN,
                "2NN": make_rank_on_knn(2),
                "3NN": make_rank_on_knn(3),
                "4NN": make_rank_on_knn(4),
                "5NN": make_rank_on_knn(5),
                "Prototype": rank_on_prototype,
                "Progenitor": rank_on_progenitor(emergence_order[0]),
                "Exemplar": make_rank_on_exemplar(s_val),
                "Exemplar (s=0.001)": make_rank_on_exemplar(0.001),
                "Exemplar (s=0.1)": make_rank_on_exemplar(0.1),
                "Exemplar (s=1)": make_rank_on_exemplar(1),
                "Local": rank_on_1NN
            }

            res = {}
            expected_wins = {}

            if measure_type == "ll":
                ll_rand = get_probability_rand(emergence_order)
                #models["Null"][filename[:-4]] = ll_rand
                print("RANDOM SCORE: ", ll_rand)

            ranking_types = ["global" if name != "Local" else "local" for name in name_to_model]
            if measure_type == "ll":
                ll_models, model_order = get_probability_score_multi(emergence_order, all_vecs, name_to_model, ranking_types)
            
                for name, ll in zip(model_order, ll_models):
                    print(" === MODEL ===")
                    print(f"{name}: {ll}")
                    diff = ll - ll_rand
                    res[name] = (diff, len(all_vecs))
                    models[name][filename[:-4]] = (diff, len(all_vecs))
                
                with open(f"{individual_path}/{filename[:-4]}.p", "wb") as p_file:
                    pickle.dump(res, p_file)
            
            elif measure_type == "rank":
                for model_name, ranking_func in name_to_model.items():
                    ranking_type = "global" if model_name != "Local" else "local"
                    expected_ratio, pred_ranks_vec = predict_seq(all_vecs, emergence_order, ranking_func, get_rank_score_avg, ranking_type=ranking_type, expected_wins=True)
                    expected_wins[model_name] = (expected_ratio, len(all_vecs))
                    models[model_name][filename[:-4]] = (expected_ratio, len(all_vecs))


                with open(f"{individual_path}/{filename[:-4]}-rank.p", "wb") as p_file:
                    pickle.dump(expected_wins, p_file)

                
            # with open(f"{individual_path}/{filename[:-4]}.p", "rb") as p_file:
            #     old_res = pickle.load(p_file)
            #     old_res.update(res)
            #with open(f"{individual_path}/{filename[:-4]}.p", "b") as p_file:
                #old_res = pickle.load(p_file)
            
            #old_res.update(res)

            
        
    #with open(out_path, "rb") as p_file:
        #old_models = pickle.load(p_file)
    
    #old_models.update(models)

    with open(out_path, "wb") as p_file:
        pickle.dump(models, p_file)

def main(domain: str, field: str, measure: str, cv: bool, shuffle: bool, shuffle_years: bool, use_individual: bool, train_test_split: bool):    
    shuffle_emergence_order = shuffle_years

    if shuffle_emergence_order == False:
        num_shuffle = 1
    else:
        num_shuffle = 3

    if domain == "turing":
        vecs_path = f"data/turing_winners/sbert-abstracts-ordered"
        order_path = vecs_path
        train_path = f"data/turing_winners/pickled-tuning-2"
        individual_path = "data/turing_winners/pickled"
        #out_path = "results/shuffle/cs/"
        out_path = "results/summary/cv/cs-2.p"
        s_vals_path = "data/turing_winners/s-vals"
        field = "cs"
    else:
        vecs_path = f"data/nobel_winners/{field}/abstracts-ordered"
        order_path = vecs_path
        train_path = f"data/nobel_winners/{field}/pickled-tuning-2"
        individual_path = f"data/nobel_winners/{field}/pickled"
        #out_path = f"results/shuffle/{field}/"
        out_path = f"results/summary/cv/{field}-2.p"
        s_vals_path = f"data/nobel_winners/{field}/s-vals"

    models = {
        "1NN": {},
        "2NN": {},
        "3NN": {},
        "4NN": {},
        "5NN": {},
        "Prototype": {},
        "Progenitor": {},
        #"Exemplar (s=1)": {},
        "Exemplar": {},
        "Local": {}
    }

    field_to_s_val = {
        "cogsci": 0.09,
        "economics": 0.18,
        "physics": 0.08,
        "medicine": 0.1,
        "chemistry": 0.11,
        "cs": 0.01
    }

    all_filenames = list(os.listdir(vecs_path))

    if cv:
        field_to_s_val_folds = {
            "cs": [0.01, 0.01, 0.01, 0.01, 0.01],
            "chemistry": [0.08, 0.09, 0.08, 0.09, 0.09],
            "economics": [0.07, 0.1, 1.0, 0.1, 0.07],
            "medicine": [0.09, 0.1, 0.1, 0.09, 0.09],
            "physics": [0.08, 0.08, 0.08, 0.08, 0.08]
        }

        fold_train_inds = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: []
        }

        all_names = [filename[:-4] for filename in all_filenames]

        for i in range(5):
            with open(f"{s_vals_path}/exemplar-grid-fold_{i}.p", "rb") as vecs_f:
                train_data = pickle.load(vecs_f)
                train_names = train_data[list(train_data.keys())[0]].keys()
                train_inds = [all_names.index(item) for item in train_names]
                fold_train_inds[i].extend(train_inds)

    elif use_individual and not shuffle:
        out_path = f"results/full/{field}.p"

        models = {
            "1NN": {},
            "2NN": {},
            "3NN": {},
            "4NN": {},
            "5NN": {},
            "Prototype": {},
            "Progenitor": {},
            "Exemplar": {},
            #"Exemplar (s=1)": {},
            #"Exemplar (s=0.001)": {},
            #"Exemplar (s=0.1)": {},
            "Local": {},
            "Null": {}
        }
        if field == "cs":
            individual_s_val_path = "data/turing_winners/individual-s-vals"
            individual_out_path = "data/turing_winners/pickled-full"
        else:
            individual_s_val_path = f"data/nobel_winners/{field}/individual-s-vals"
            individual_out_path = f"data/nobel_winners/{field}/pickled-full"
            
        for i, filename in enumerate(os.listdir(individual_s_val_path)):
            if filename.endswith(".p"):
                print(filename)

            res = {}
            scientist_name = filename[:-2]
            csv_name = f"{scientist_name}.csv"
            vecs_filename = os.path.join(vecs_path, csv_name)
            order_filename = os.path.join(order_path, csv_name)
            out_filename = os.path.join(individual_out_path, filename)
                        

            all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
            emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)

            if os.path.exists(out_filename):
                with open(out_filename, "rb") as done_f:
                    done_models = pickle.load(done_f)
                    for model in done_models:
                        models[model][scientist_name] = done_models[model]
                ll_rand = get_probability_rand(emergence_order)
                models["Null"][scientist_name] = ll_rand
                print("Already done this scientist")
                continue

            scientist_s_val_path = os.path.join(individual_s_val_path, filename)
            with open(scientist_s_val_path, "rb") as s_file:
                try:
                    s_val = float(pickle.load(s_file)["s"])
                except:
                    print("Could not read s-val file")
                    continue

            name_to_model = {
                "1NN": rank_on_1NN,
                "2NN": make_rank_on_knn(2),
                "3NN": make_rank_on_knn(3),
                "4NN": make_rank_on_knn(4),
                "5NN": make_rank_on_knn(5),
                "Prototype": rank_on_prototype,
                "Progenitor": rank_on_progenitor(emergence_order[0]),
                "Exemplar": make_rank_on_exemplar(s_val),
                #"Exemplar (s=0.001)": make_rank_on_exemplar(0.001),
                #"Exemplar (s=0.1)": make_rank_on_exemplar(0.1),
                #"Exemplar (s=1)": make_rank_on_exemplar(1),
                "Local": rank_on_1NN
            }

            res = {}
            expected_wins = {}

            if measure == "ll":
                ll_rand = get_probability_rand(emergence_order)
                models["Null"][scientist_name] = ll_rand
                print("RANDOM SCORE: ", ll_rand)

            ranking_types = ["global" if name != "Local" else "local" for name in name_to_model]

            if measure == "ll":
                ll_models, model_order = get_probability_score_multi(emergence_order, all_vecs, name_to_model, ranking_types)
            
                for name, ll in zip(model_order, ll_models):
                    print(" === MODEL ===")
                    print(f"{name}: {ll}")
                    diff = ll - ll_rand
                    res[name] = (diff, len(all_vecs))
                    models[name][scientist_name] = (diff, len(all_vecs))

                with open(f"{individual_out_path}/{scientist_name}.p", "wb") as p_file:
                    pickle.dump(res, p_file)

            elif measure == "rank":
                for model_name, ranking_func in name_to_model.items():
                    ranking_type = "global" if model_name != "Local" else "local"
                    expected_ratio, pred_ranks_vec = predict_seq(all_vecs, emergence_order, ranking_func, get_rank_score_avg, ranking_type=ranking_type, expected_wins=True)
                    expected_wins[model_name] = (expected_ratio, len(all_vecs))
                    models[model_name][filename[:-4]] = (expected_ratio, len(all_vecs))

                with open(f"{individual_path}/{filename[:-4]}-rank.p", "wb") as p_file:
                    pickle.dump(expected_wins, p_file)
        
        with open(out_path, "wb") as p_file:
            pickle.dump(models, p_file)
        return


    elif train_test_split:
        selected = [name[:-2] + ".csv" for name in os.listdir(train_path) if name.endswith(".p")]
        train_inds = [all_filenames.index(item) for item in selected if item.endswith(".csv")]

    elif shuffle and measure == "rank":
        models = {
            "1NN": {},
            "Prototype": {},
            "Progenitor": {},
            "Exemplar": {},
            "Local": {},
            "Null": {}
        }

        out_path = f"results/full-rank/{field}.p"
        for i, filename in enumerate(all_filenames): 
            if filename.endswith(".csv"):
                print(filename)
            
            res = {}
            
            scientist_p_filename = f"{filename[:-4]}.p"
            vecs_filename = os.path.join(vecs_path, filename)
            order_filename = os.path.join(order_path, filename)
            if field == "cs":
                individual_s_val_path = "data/turing_winners/individual-s-vals"
                individual_out_path = "data/turing_winners/pickled-full"
            else:
                individual_s_val_path = f"data/nobel_winners/{field}/individual-s-vals"
                individual_out_path = f"data/nobel_winners/{field}/pickled-full"

            all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
            if len(all_vecs) < 5:
                continue
            emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)
            scientist_s_val_path = os.path.join(individual_s_val_path, scientist_p_filename)
            with open(scientist_s_val_path, "rb") as s_file:
                try:
                    s_val = float(pickle.load(s_file)["s"])
                except:
                    print("Could not read s-val file")
                    continue
    
            name_to_model = {
                "1NN": rank_on_1NN,
                "Prototype": rank_on_prototype,
                "Progenitor": rank_on_progenitor(emergence_order[0]),
                "Exemplar": make_rank_on_exemplar(s_val),
                "Local": rank_on_1NN
            }

            model_order = {i: model_name for i, model_name in enumerate(name_to_model.keys())}

            results = shuffle_test_expected_vals(num_shuffle, name_to_model, model_order, vecs_filename, order_filename)
            
            for i, model_name in model_order.items():
                res[model_name] = {"p-val": results[0][i], "win_ratio": results[1][i], "yerr": results[2][i], "tval": results[3][i], "num_papers": len(all_vecs)}
                models[model_name][filename[:-4]] = res[model_name]

            with open(f"{individual_out_path}/{filename[:-4]}-rank.p", "wb") as p_file:
                pickle.dump(res, p_file)

        with open(out_path, "wb") as p_file:
            pickle.dump(models, p_file)

        return


    elif shuffle and measure == "ll":
        models["Null"] = {}
        for s_n in range(num_shuffle):
            if field == "cs":
                individual_path = "data/turing_winners/pickled-shuffle"
                out_path = f"results/summary/shuffle-ll/cs/cs-{s_n}.p"
                individual_s_val_path = "data/turing_winners/individual-s-vals"
            else:
                individual_path = f"data/nobel_winners/{field}/pickled-shuffle"
                out_path = f"results/summary/shuffle-ll/{field}/{field}-{s_n}.p"
                individual_s_val_path = f"data/nobel_winners/{field}/individual-s-vals"

            for i, filename in enumerate(all_filenames): 
                if filename.endswith(".csv"):
                    print(filename)
                if train_test_split and i in train_inds:
                    continue
                vecs_filename = os.path.join(vecs_path, filename)
                order_filename = os.path.join(order_path, filename)
                all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
                emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)
                all_vecs, emergence_order = shuffle_emergence_years(all_vecs, emergence_order)
                
                if use_individual:
                    scientist_s_val_path = os.path.join(individual_s_val_path, f"{filename[:-4]}.p")
                    if not os.path.exists(scientist_s_val_path):
                        s_val = field_to_s_val[field]
                    else:
                        with open(scientist_s_val_path, "rb") as s_file:
                            try:
                                s_val = float(pickle.load(s_file)["s"])
                            except:
                                print("Could not read s-val file")
                                continue
                else:
                    s_val = field_to_s_val[field]

                name_to_model = {
                    "1NN": rank_on_1NN,
                    #"2NN": make_rank_on_knn(2),
                    #"3NN": make_rank_on_knn(3),
                    #"4NN": make_rank_on_knn(4),
                    #"5NN": make_rank_on_knn(5),
                    "Prototype": rank_on_prototype,
                    "Progenitor": rank_on_progenitor(emergence_order[0]),
                    #"Exemplar (s=1)": rank_on_exemplar,
                    "Exemplar": make_rank_on_exemplar(field_to_s_val[field]),
                    "Local": rank_on_1NN
                }

                res = {}
                ll_rand = get_probability_rand(emergence_order)
                print("RANDOM SCORE: ", ll_rand)
                models["Null"][filename[:-4]] = ll_rand
                res["Null"] = ll_rand

                ranking_types = ["global" if name != "Local" else "local" for name in name_to_model]
                ll_models, model_order = get_probability_score_multi(emergence_order, all_vecs, name_to_model, ranking_types)
            
                for name, ll in zip(model_order, ll_models):
                    print(" === MODEL ===")
                    print(f"{name}: {ll}")
                    diff = ll - ll_rand
                    res[name] = (diff, len(all_vecs))
                    
                    
                    models[name][filename[:-4]] = (diff, len(all_vecs))
                   
                    
                # with open(f"{individual_path}/{filename[:-4]}.p", "rb") as p_file:
                #     old_res = pickle.load(p_file)
                #     old_res.update(res)

                with open(f"{individual_path}/{filename[:-4]}-{s_n}.p", "wb") as p_file:
                    pickle.dump(res, p_file)
            
            with open(out_path, "wb") as p_file:
                pickle.dump(models, p_file)
        
        return

    if cv:
        for fold in range(5):
            out_path = f"results/cv5/{field}-{fold}.p"
            if field == "cs":
                individual_path = f"data/turing_winners/pickled-cv5/{fold}"
            else:
                individual_path =  f"data/nobel_winners/{field}/pickled-cv5/{fold}"
            evaluate_on_files(measure, True, False, all_filenames, vecs_path, order_path, train_path, individual_path, out_path, s_vals_path, field, field_to_s_val_folds[field][fold], fold_train_inds[fold])
    
    else: 
        out_path = f"results/full/{field}.p"
        if field == "cs":
            individual_path = "data/turing_winners/pickled-full/"
        else:
            individual_path =  f"data/nobel_winners/{field}/pickled-full/"

        # use same s-value just to check how well it works across domains (no train/test split)
        s_val = 0.01

        evaluate_on_files(measure, True, False, all_filenames, vecs_path, order_path, None, individual_path, out_path, None, field, s_val, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "chemistry", "medicine", "economics", "cogsci"])
    parser.add_argument("--measure", help="use rank-based measure or log-likelihood based measure", choices=["rank", "ll"], required=True)
    parser.add_argument("-c", help="use 5-fold cross-validated inputs or not", action="store_true")
    parser.add_argument("-s", help="shuffle or not", action="store_true")
    parser.add_argument("--sy", help="shuffle years of paper publication or not", action="store_true")
    parser.add_argument("-i", help="use individual kernel widths or not", action="store_true")
    parser.add_argument("-t", help="use train-test split or not", action="store_true")
    args = parser.parse_args()

    main(args.type, args.field, args.measure, args.c, args.s, args.sy, args.i, args.t)
