import argparse
from hyperopt import hp
from hyperopt import tpe, fmin, STATUS_OK
import os
import pickle
from typing import List, Callable

from src.models.predict import make_rank_on_exemplar, get_probability_score, get_attested_order, get_emergence_order

def make_objective(all_vecs: List, emergence_order: dict) -> Callable:
    def objective(s: float) -> float:
        model = make_rank_on_exemplar(s)
        score = get_probability_score(emergence_order, all_vecs, model, return_log_L_only=True, suppress_print=True)
        loss = -score
        return loss
    
    return objective


def optimize_one_scientist(all_vecs: List, emergence_order: dict) -> float:
    space = hp.uniform("s", 0, 1)
    to_min = make_objective(all_vecs, emergence_order)
    best = fmin(fn=to_min, space=space, algo=tpe.suggest, max_evals=100)

    return best



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "chemistry", "medicine", "economics", "cogsci"])
    args = parser.parse_args()

    if args.type == "nobel":
        vecs_path = f"data/nobel_winners/{args.field}/abstracts-ordered"
        individual_path = f"data/nobel_winners/{args.field}/individual-s-vals"
        results_path = f"results/summary/s_opt/{args.field}.p"
    else:
        vecs_path = "data/turing_winners/sbert-abstracts-ordered"
        individual_path = "data/turing_winners/individual-s-vals"
        results_path = "results/summary/s_opt/cs.p"

    best_s_vals = {}
    for i, filename in enumerate(os.listdir(vecs_path)):
        if filename.endswith(".csv"):
            print(i, filename)

        if i > 20:
            assert False

        vecs_filename = os.path.join(vecs_path, filename)
        order_filename = os.path.join(vecs_path, filename)
        individual_filename = os.path.join(individual_path, filename)

        all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
        emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)

        if len(all_vecs) < 5:
            continue

        best = optimize_one_scientist(all_vecs, emergence_order)
        print(best)
        best_s_vals[filename[:-4]] = best

        with open(f"{individual_path}/{filename[:-4]}.p", "wb") as ind_file:
            pickle.dump(best, ind_file)

    
    with open(f"{results_path}/{filename[:-4]}.p", "wb") as res_file:
        pickle.dump(best_s_vals, res_file)


