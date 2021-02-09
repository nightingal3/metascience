import argparse
from functools import reduce
import numpy as np
import os
from random import shuffle
import pdb
import pickle
from typing import List

from src.models.predict import make_rank_on_exemplar, get_emergence_order, get_attested_order, get_probability_score_multi, get_probability_rand

def train_test_split_inds(num_inds: int, percent_train: float) -> tuple:
    inds = list(range(num_inds))
    num_train = int(percent_train * num_inds)
    shuffle(inds)

    train = inds[:num_train + 1]
    test = inds[num_train + 1:]
    return train, test

def cv_split_inds(num_inds: int, k: int) -> List:
    inds = list(range(num_inds))
    if num_inds == 0 or num_inds == 1:
        return inds
    shuffle(inds)

    return [list(x) for x in np.array_split(inds, k)]


def test_s_vals(domain: str, field: str) -> None:
    if domain == "turing":
        out_path = "data/turing_winners/s-vals"
        vecs_path = "data/turing_winners/sbert-abstracts-ordered"
        tuning_path = "data/turing_winners/pickled-tuning/"
        tuning_path_2 = "data/turing_winners/pickled-tuning-2/"
    else:
        out_path = f"data/nobel_winners/{field}/s-vals"
        vecs_path = f"data/nobel_winners/{field}/abstracts-ordered"
        tuning_path = f"data/nobel_winners/{field}/pickled-tuning/"
        tuning_path_2 = f"data/nobel_winners/{field}/pickled-tuning-2/"

    order_path = vecs_path
    from_previous_inds = False

    #s_vals = [0.001 * i for i in range(1, 101)]
    s_vals = [0.01 * i for i in range(1, 101)]
    models = {val: {} for val in s_vals}
    num_samples = len([name for name in os.listdir(vecs_path) if name.endswith(".csv")])

    if not from_previous_inds:
        train_inds, test_inds = train_test_split_inds(num_samples, 0.5)
    else:
        train_inds = []
        with open(inds_path + f"train_inds_{i}.txt", "r") as f:
            for row in f.readlines():
                train_inds.append(int(row))

        selected = [name[:-2] + ".csv" for name in os.listdir(tuning_path) if name.endswith(".p")]
        all_names = list(os.listdir(vecs_path))
        train_inds = [all_names.index(item) for item in selected if item.endswith(".csv")]


    for i, filename in enumerate(os.listdir(vecs_path)):  
        if filename.endswith(".csv"):
            print(i, filename)
            if i in test_inds:
                continue
            vecs_filename = os.path.join(vecs_path, filename)
            order_filename = os.path.join(order_path, filename)
            all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
            emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)

            name_to_model = {val: make_rank_on_exemplar(val) for val in s_vals}

            res = {}
            ll_rand = get_probability_rand(emergence_order)
            print("RANDOM SCORE: ", ll_rand)

            # for name in name_to_model:
            #     print(name)
            #     ranking_type = "local" if name == "Local" else "global"
            #     ll_model, _, _ = get_probability_score(emergence_order, all_vecs, name_to_model[name], ranking_type=ranking_type)
            #     print("MODEL SCORE:", ll_model)
            #     diff = ll_model - ll_rand

            #     print(f"LL ratio {name}: ", diff)
            #     models[name][filename[:-4]] = (diff, len(all_vecs))
            #     res[name] = (diff, len(all_vecs))

            ranking_types = ["global"] * len(s_vals)
            ll_models, model_order = get_probability_score_multi(emergence_order, all_vecs, name_to_model, ranking_types)
            
            for name, ll in zip(model_order, ll_models):
                print(" === MODEL ===")
                print(f"{name}: {ll}")
                diff = ll - ll_rand
                res[name] = (diff, len(all_vecs))
                models[name][filename[:-4]] = (diff, len(all_vecs))

            with open(f"{tuning_path_2}{filename[:-4]}.p", "rb") as p_file:
                old_info = pickle.load(p_file)
                old_info.update(res)
                print(old_info)
            
            with open(f"{tuning_path_2}{filename[:-4]}.p", "wb") as p_file:
                pickle.dump(old_info, p_file)

            assert False

    with open(f"{out_path}/exemplar-grid.p", "rb") as p_file:
        old_model = pickle.load(p_file)
        old_model.update(models)
        print(old_model)
        assert False
    with open(f"{out_path}/exemplar-grid.p", "wb") as p_file:
        pickle.dump(models, p_file)


    print("train inds")
    print(train_inds)
   
def run_cv_s_vals(domain: str, field: str, cv: int) -> None:
    if domain == "turing":
        out_path = "data/turing_winners/s-vals"
        vecs_path = "data/turing_winners/sbert-abstracts-ordered"
        tuning_path = "data/turing_winners/pickled-tuning/"
        tuning_path_2 = "data/turing_winners/pickled-tuning-2/"
    else:
        out_path = f"data/nobel_winners/{field}/s-vals"
        vecs_path = f"data/nobel_winners/{field}/abstracts-ordered"
        tuning_path = f"data/nobel_winners/{field}/pickled-tuning/"
        tuning_path_2 = f"data/nobel_winners/{field}/pickled-tuning-2/"

    order_path = vecs_path
    from_previous_inds = False

    #s_vals = [0.001 * i for i in range(1, 101)]
    #s_vals = [0.01 * i for i in range(1, 101)]
    #s_vals = [0.001 * i for i in range(1, 101)]
    s_vals = [0.001 * i for i in range(1, 3)]
    num_samples = len([name for name in os.listdir(vecs_path) if name.endswith(".csv")])
    cv_folds = cv_split_inds(num_samples, cv)
    
    with open(f"{out_path}/folds.txt", "w") as inds_f:
        for fold in cv_folds:
            inds_f.write(str(fold) + "\n")


    for fold in range(len(cv_folds)):
        print(f"Fold {fold}")
        models = {val: {} for val in s_vals}
        test_inds = cv_folds[fold]

        for i, filename in enumerate(os.listdir(vecs_path)):
            if filename.endswith(".csv"):
                print(i, filename)
            if i in test_inds:
                continue
            
            ll_diff = {}
            res = {}

            vecs_filename = os.path.join(vecs_path, filename)
            order_filename = os.path.join(order_path, filename)
            all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
            emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)

            name_to_model = {val: make_rank_on_exemplar(val) for val in s_vals}

            ll_rand = get_probability_rand(emergence_order)
            print("RANDOM SCORE: ", ll_rand)

            ranking_types = ["global"] * len(s_vals)
            ll_models, model_order = get_probability_score_multi(emergence_order, all_vecs, name_to_model, ranking_types)
            
            for name, ll in zip(model_order, ll_models):
                print(" === MODEL ===")
                print(f"{name}: {ll}")
                diff = ll - ll_rand
                if name in ll_diff:
                    ll_diff[name].append(diff)
                else:
                    ll_diff[name] = [diff]
            
            for name, ll in zip(model_order, ll_models):
                res[name] = sum(ll_diff[name])/len(ll_diff[name])
                if name in models and filename[:-4] in models[name]:
                    models[name][filename[:-4]].append(res[name])
                else:
                    models[name][filename[:-4]] = res[name]

            with open(f"{tuning_path}{filename[:-4]}_{fold}.p", "wb") as p_file:
                pickle.dump(models, p_file)
        

        with open(f"{out_path}/exemplar-grid-fold_{fold}.p", "wb") as p_file:
            pickle.dump(models, p_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "chemistry", "medicine", "economics", "cogsci"])
    parser.add_argument("-c", "--cv", help="run cross validation with c folds", type=int)
    args = parser.parse_args()

    if args.cv is not None:
        run_cv_s_vals(args.type, args.field, args.cv)
    else:
        test_s_vals(args.type, args.field)