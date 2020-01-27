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
from scipy.special import softmax
from sklearn import preprocessing, metrics
import statsmodels.stats.api as sms
import pdb
import os
import pickle
from pprint import pprint

def get_attested_order(vecs_filename: str, vecs_col: int = 1, label: bool = False) -> List:
    rev_order = []
    with open(vecs_filename, "r") as vecs_file:
        reader = csv.reader(vecs_file)
        for row in reader:
            if not label:
                rev_order.append(ast.literal_eval(row[vecs_col]))
            else:
                rev_order.append(row[vecs_col])
    return rev_order[::-1]


def rank_on_1NN(emerged: List, unemerged: List, keep_sim: bool = False, dist="cosine") -> List:
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
    def rank_on_knn(emerged: List, unemerged: List, keep_sim: bool = False, dist="cosine") -> List:
        nonlocal curr_iter
        nonlocal emerged_num

        if len(emerged) != emerged_num: # A timestep has gone by
            emerged_num = len(emerged)
            curr_iter += 1

        closest = {tuple(vec): 0 for vec in unemerged}
        curr_k = min(k, curr_iter)

        for candidate_vec in unemerged:
            sim_sum = 0
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
def rank_on_prototype(emerged: List, unemerged: List, keep_sim: bool = False, dist="cosine") -> List:
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


def rank_on_progenitor(progenitor: tuple) -> Callable:
    def rank(emerged: List, unemerged: List, keep_sim: bool = False, dist="cosine") -> List:
        closest = {tuple(vec): 0 for vec in unemerged}
        for vec in unemerged:
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

def rank_on_exemplar(emerged: List, unemerged: List, keep_sim: bool = False, dist="cosine") -> List:
    sum_sim = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sum_sim[tuple(candidate_vec)] += get_sim(emerged_vec, candidate_vec, dist=dist)[0][0]

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


def get_sim(vec_1: List, vec_2: List, dist="cosine") -> float:
    if dist == "cosine":
        return metrics.pairwise.cosine_similarity(
            np.asarray(vec_1).reshape(1, -1), np.asarray(vec_2).reshape(1, -1)
        )
    elif dist == "euclidean":
        return 1/(1 + metrics.pairwise.euclidean_distances(
            np.asarray(vec_1).reshape(1, -1), np.asarray(vec_2).reshape(1, -1)
        ))


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



def get_probability_score(emergence_order: dict, all_vecs: List, labels: List, ranking_func: Callable, ranking_type: str = "global", carry_error: bool = False) -> float:
    last_timestep = max(emergence_order.keys())
    log_L = 0
    emerged_papers = []
    tails = []

    for t in range(last_timestep):
        #print("TIMESTEP: ", t)
        curr_papers = emergence_order[t]
        print("curr papers: ", len(curr_papers))
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

        #pdb.set_trace()
        pred_and_sim = ranking_func(last_papers, available_papers, keep_sim=True)
        pred, sim = zip(*pred_and_sim)
        print(len(pred))
        sim_softmax = [prob / sum(sim) for prob in sim]
        if not carry_error:
            for vec in next_papers:
                found_index = pred.index(vec)
                #print(f"{found_index}: {sim_softmax[found_index]}")                
                log_L += np.log(sim_softmax[found_index])
        
        if carry_error:
            for vec in pred[:len(next_papers)]:
                closest_ind = np.argpartition(distance.cdist(np.array([vec]), np.asarray(emerged_papers), metric="cosine"), 0)[0][0]
                tails.append(closest_ind)
                emerged_papers.append(vec)


    return log_L, emerged_papers, tails

def get_probability_rand(emergence_order: dict) -> float:
    """ Returns a fixed probability for a predicted sequence based on the number
    papers emerging at each timestep. 
    timesteps: [run_0, run_1, run_2...]

    *** This may not be correct mathematically (!)
    """
    timesteps = make_timesteps(emergence_order)
    log_L = 0
    denom = sum(timesteps)
    for timestep in timesteps:
        for i in range(timestep):
            log_L += np.log(1 / denom)
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


def shuffle_test_multi(n_iter: int, target_val_list: List, emergence_order: dict, vecs_filename: str, order_filename: str, return_raw_counts: bool = False) -> tuple:
    cumulative_rank_diffs = []
    trial_timestep_rank_diff = None
    _attested_order = get_attested_order(vecs_filename)
    _emergence_order = get_emergence_order(order_filename)
    pvals = []
    avg_rank_diffs = []
    avg_rank_diff_timesteps_multi = []
    upper_conf_intervals = []
    lower_conf_intervals = []

    for target_val in target_val_list:
        higher = 0
        lower = 0
        for i in range(n_iter):
            print(i)
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
        upper_conf_intervals.append(upper_conf_interval)
        lower_conf_intervals.append(lower_conf_interval)

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


if __name__ == "__main__":
    vecs_path = "data/turing_winners/vecs-abstracts-only"
    order_path = "data/turing_winners/vecs-abstracts-ordered"
    models = {
        "1NN": {},
        "2NN": {},
        "3NN": {},
        "4NN": {},
        "5NN": {},
        "Prototype": {},
        "Progenitor": {},
        "Exemplar": {},
        "Local": {}
        }

    all_vecs = get_attested_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", vecs_col=2)
    labels = get_attested_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", vecs_col=1, label=True)
    emergence_order = get_emergence_order("data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final-ordered.csv", vecs_col=2)
    _, ranks, tails = get_probability_score(emergence_order, all_vecs, labels, rank_on_1NN, ranking_type="global", carry_error=True)
    inds = [all_vecs.index(list(vec)) for vec in ranks]
    titles = [labels[all_vecs.index(list(vec))] for vec in ranks]

    pprint(titles)
    assert False
    for filename in os.listdir(vecs_path):
        if filename.endswith(".csv"):
            print(filename)
            vecs_filename = os.path.join(vecs_path, filename)
            order_filename = os.path.join(order_path, filename)
            all_vecs = get_attested_order(vecs_filename)
            emergence_order = get_emergence_order(order_filename)

            name_to_model = {
                "1NN": rank_on_1NN,
                "2NN": make_rank_on_knn(2),
                "3NN": make_rank_on_knn(3),
                "4NN": make_rank_on_knn(4),
                "5NN": make_rank_on_knn(5),
                "Prototype": rank_on_prototype,
                "Progenitor": rank_on_progenitor(all_vecs[0]),
                "Exemplar": rank_on_exemplar,
                "Local": rank_on_1NN
            }

            for name in name_to_model:
                print(name)
                ranking_type = "local" if name == "Local" else "global"
                ll_model = get_probability_score(emergence_order, all_vecs, name_to_model[name], ranking_type=ranking_type)
                ll_rand = get_probability_rand(emergence_order)
                diff = ll_model - ll_rand

                print(f"LL ratio {name}: ", diff)
                models[name][filename[:-4]] = diff
           
    with open("model-LL.p", "wb") as p_file:
        pickle.dump(models, p_file)
    assert False

    avg_cumulative_errs = []
    rank_on_2NN = make_rank_on_knn(2)

    name_to_func = {
                "2NN": rank_on_2NN,
                "1NN": rank_on_1NN,
                "Prototype": rank_on_prototype,
                #"Progenitor": rank_on_progenitor(all_vecs[0]),
                #"Exemplar": rank_on_exemplar,
                #"Local": rank_on_1NN,
                #"CRP": crp_model.rank_on_clusters
            }

    for model in name_to_func:
        print("MODEL: ",  model)
        model_type = "local" if model == "Local" else "global"
        error, rank_diff_per_timestep = predict_seq(
            all_vecs,
            emergence_order,
            name_to_func[model],
            get_rank_score_best,
            ranking_type=model_type,
        )
        print("error: ", error)
        print("rank diff per timestep: ", rank_diff_per_timestep)
        avg_cumulative_errs.append(error)

    print(shuffle_test_multi(n_iter=1000, target_val_list=avg_cumulative_errs, emergence_order=emergence_order, vecs_filename=vecs_filename, order_filename=order_filename))
