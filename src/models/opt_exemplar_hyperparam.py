import argparse
from hyperopt import hp
from hyperopt import tpe, fmin, STATUS_OK
from hyperopt.pyll import scope
import os
import pickle
from typing import List, Callable

from src.models.predict import make_rank_on_exemplar, make_rank_on_knn, make_rank_on_prototype, rank_on_progenitor, make_rank_on_1NN, get_probability_score, get_probability_score_crp, get_attested_order, get_emergence_order
from src.models.chinese_restaurant import CRP

def make_objective(all_vecs: List, emergence_order: dict, model_type: str) -> Callable:
    def objective(s: float) -> float:
        ranking_type = "global"
        if model_type == "exemplar":
            model = make_rank_on_exemplar(s)
        elif model_type == "kNN":
            model = make_rank_on_knn(s)
        elif model_type == "prototype":
            model = make_rank_on_prototype(s)
        elif model_type == "progenitor":
            model = rank_on_progenitor(emergence_order[0], s=s)
        elif model_type == "local":
            model = make_rank_on_1NN(s=s)
            ranking_type = "local"
        elif model_type == "crp":
            model = CRP(alpha=s, starting_points=emergence_order[0], use_pca=False, clustering_type="kmeans")

        if model_type == "crp":
            score = get_probability_score_crp(emergence_order, all_vecs, model, return_log_L_only=True, suppress_print=True)
        else:
            score = get_probability_score(emergence_order, all_vecs, model, ranking_type=ranking_type, return_log_L_only=True, suppress_print=True)
        loss = -score
        return loss
    
    return objective

def make_objective_crp(all_vecs: List, emergence_order: dict, model_type: str) -> Callable:
    def objective(params: dict) -> float:
        alpha = params["alpha"]
        s = params["s"]
        likelihood_type = model_type[:-4]
        model = CRP(alpha=alpha, starting_points=emergence_order[0], likelihood_type=likelihood_type, likelihood_hyperparam=s, use_pca=False, clustering_type="kmeans")
        score = get_probability_score_crp(emergence_order, all_vecs, model, return_log_L_only=True, suppress_print=True)
        
        return -score
    
    return objective

def optimize_one_scientist(all_vecs: List, emergence_order: dict, model_type: str) -> dict:
    if model_type == "crp":
        space = hp.uniform("s", 0, 100)
    else:
        space = hp.uniform("s", 0, 1)
    to_min = make_objective(all_vecs, emergence_order, model_type=model_type)
    best = fmin(fn=to_min, space=space, algo=tpe.suggest, max_evals=100)

    return best

def optimize_one_scientist_crp(all_vecs: List, emergence_order: dict, model_type: str) -> dict:
    space = {
        "alpha": hp.uniform("alpha", 0, 100),
        "s": hp.uniform("s", 0, 1)
    }

    to_min = make_objective_crp(all_vecs, emergence_order, model_type=model_type)
    best = fmin(fn=to_min, space=space, algo=tpe.suggest, max_evals=100)

    return best

def grid_search_knn(lower_bound: int, upper_bound: int, all_vecs: List, emergence_order: dict) -> dict:
    best = {}
    upper_lim = min(upper_bound + 1, len(all_vecs))
    for k in range(lower_bound, upper_lim):
        model = make_rank_on_knn(k)
        score = get_probability_score(emergence_order, all_vecs, model, ranking_type="global", return_log_L_only=True, suppress_print=True)
        best[k] = score
    
    return {"s": max(best.items(), key=lambda x: x[1])[0]}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("--model_type", help="which model type to optimize", choices=["exemplar", "kNN", "progenitor", "prototype", "local", "exemplar-crp", "progenitor-crp", "prototype-crp", "local-crp"], required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "physics-random", "chemistry", "chemistry-random", "medicine", "medicine-random", 
    "economics", "economics-random", "cs-random"])
    args = parser.parse_args()

    if args.type == "nobel":
        #vecs_path = f"data/nobel_winners/{args.field}/abstracts-ordered"
        vecs_path = f"data/nobel_winners/{args.field}/abstracts-ordered"
        individual_path = f"data/nobel_winners/{args.field}/individual-s-vals/{args.model_type}"
        results_path = f"results/summary/s_opt"
    else:
        vecs_path = "data/turing_winners/sbert-abstracts-ordered"
        individual_path = f"data/turing_winners/individual-s-vals/{args.model_type}"
        results_path = f"results/summary/s_opt"

    best_s_vals = {}
    for i, filename in enumerate(os.listdir(vecs_path)):
        if filename.endswith(".csv"):
            print(i, filename)

        vecs_filename = os.path.join(vecs_path, filename)
        order_filename = os.path.join(vecs_path, filename)
        individual_filename = os.path.join(individual_path, filename)

        all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
        emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)

        if len(all_vecs) < 5:
            continue
        if args.model_type == "kNN":
            best = grid_search_knn(1, 20, all_vecs, emergence_order)
        if "crp" in args.model_type:
            best = optimize_one_scientist_crp(all_vecs, emergence_order, args.model_type)
        else:
            best = optimize_one_scientist(all_vecs, emergence_order, args.model_type)
        print(best)
        best_s_vals[filename[:-4]] = best

        with open(f"{individual_path}/{filename[:-4]}-{args.model_type}.p", "wb") as ind_file:
            pickle.dump(best, ind_file)

    
    with open(f"{results_path}/{args.field}-{args.model_type}.p", "wb") as res_file:
        pickle.dump(best_s_vals, res_file)


