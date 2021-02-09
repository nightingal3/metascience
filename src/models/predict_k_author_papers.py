import ast
import argparse
import numpy as np
import os
import pandas as pd
import pdb
import pickle
from scipy.special import softmax
from typing import Callable, List

from predict import rank_on_1NN, make_rank_on_knn, get_probability_rand, rank_on_prototype, rank_on_progenitor

# k - max number of authors on a paper to consider, set k = -1 for first author only 
def get_probability_score_k_authors(k: int, vecs_and_author_info: pd.DataFrame, emergence_order: dict, all_vecs: List, all_titles: List, ranking_funcs: dict, ranking_types: dict) -> tuple:
    last_timestep = max(vecs_and_author_info["timestep"])
    log_Ls = [0] * len(ranking_funcs)
    emerged_papers = [[] for i in range(len(ranking_funcs))]
    emerged_titles = [[] for i in range(len(ranking_funcs))]

    order = []

    print("TOTAL PAPERS: ", len(all_vecs))
    for t in range(last_timestep):
        print("TIMESTEP: ", t)
        t_curr_papers = None
        t_next_papers = None
        for i, name in enumerate(ranking_funcs):
            #print("ranking: ", name)
            curr_titles = emergence_order[t]
            curr_rows = vecs_and_author_info.loc[vecs_and_author_info["title"].isin(curr_titles)]
            curr_papers = []
            for row in curr_rows.iterrows():
                curr_papers.append(tuple(ast.literal_eval(row[1]["embedding"])))
            if t_curr_papers is None:
                t_curr_papers = curr_papers
            #curr_papers = vecs_and_author_info.loc[vecs_and_author_info["year"] == t]
            #print("curr papers: ", len(curr_papers))
            next_titles = emergence_order[t + 1]
            next_rows = vecs_and_author_info.loc[vecs_and_author_info["title"].isin(next_titles)]
            next_papers = []
           
            for row in next_rows.iterrows():
                if k > 0 and row[1]["num_authors"] > k:
                    continue
                elif k == -1 and row[1]["first_author"] == 0:
                    continue
                next_papers.append(tuple(ast.literal_eval(row[1]["embedding"])))
            if t_next_papers is None:
                t_next_papers = next_papers

            emerged_papers[i].extend(t_curr_papers)
            emerged_titles[i].extend(curr_titles)
            #print("emerged papers: ", curr_titles)
            last_papers = emerged_papers[i]
            if ranking_types[name] == "local":
                last_papers = t_curr_papers

            if len(next_papers) == 0:
                continue
            
            # available_papers = all_vecs[:] #available papers should be only <= k author papers
            # for paper in emerged_papers[i]:
            #     if paper not in available_papers: # > k author paper emerged
            #         continue
            #     available_papers.remove(paper)  # Not using list comprehension so duplicates are preserved

            
            available_titles = all_titles[:] #available papers should be only <= k author papers
            for paper in emerged_titles[i]:
                if paper not in available_titles: # > k author paper emerged
                    continue
                #print("paper removed: ", paper)
                available_titles.remove(paper)  # Not using list comprehension so duplicates are preserved
            
            available_papers = []
            avail_papers = vecs_and_author_info.loc[vecs_and_author_info["title"].isin(available_titles)]
            for row in avail_papers.iterrows():
                available_papers.append(tuple(ast.literal_eval(row[1]["embedding"])))
            
            if len(available_papers) == 0: # no more <= k author papers to predict
                return log_Ls, order

            pred_and_sim = ranking_funcs[name](last_papers, available_papers, keep_sim=True)
            pred, sim = zip(*pred_and_sim)
            log_L = 0
            
            #sim_softmax = [prob / sum(sim) for prob in sim]
            sim_softmax = softmax(sim)
            try:
                next_indices = [pred.index(v) for v in t_next_papers]
            except:
                pdb.set_trace()

            for vec in t_next_papers:
                try:
                    found_index = pred.index(vec)
                except:
                    pdb.set_trace()

                next_indices = [i for i in next_indices if i != found_index]
                sim_others_excluded = [prob if i not in next_indices else 0 for i, prob in enumerate(sim_softmax)]
                sim_others_excluded = [prob/sum(sim_others_excluded) for prob in sim_others_excluded]
                #print(f"{found_index}: {sim_others_excluded[found_index]}, len: {len([i for i in sim_others_excluded if i != 0])}") 
                #if sim_others_excluded[found_index] == 0:
                    #pdb.set_trace()
                log_L += np.log(sim_others_excluded[found_index])

            log_Ls[i] += log_L
            order.append(name)
    
    return log_Ls, order

def filter_k_authors_df(k: int, vecs_and_author_info: pd.DataFrame) -> pd.DataFrame:
    if k > 0:
        filtered = vecs_and_author_info.loc[vecs_and_author_info["num_authors"] <= k]
    elif k <= 0:
        filtered = vecs_and_author_info.loc[vecs_and_author_info["first_author"] == 1]

    filtered = filtered.sort_values(by="timestep", ascending=False)
    all_timesteps = sorted(filtered.groupby(filtered["timestep"]).groups.keys())

    timesteps_rep = {all_timesteps[i]: i for i in range(len(all_timesteps))}
    filtered["timestep"] = filtered["timestep"].replace(timesteps_rep)

    return filtered

def get_random_score_k_authors(k: int, vecs_and_author_info: pd.DataFrame) -> float:
    filtered = filter_k_authors_df(k, vecs_and_author_info)
    new_emergence_order = get_emergence_order_df(filtered)
    return get_probability_rand(new_emergence_order)

def get_emergence_order_df(vecs_and_author_info: pd.DataFrame) -> dict:
    emergence_order = {}

    for row in vecs_and_author_info.iterrows():
        if row[1]["timestep"] not in emergence_order: 
            emergence_order[row[1]["timestep"]] = [row[1]["title"]]
        else:
            emergence_order[row[1]["timestep"]].append(row[1]["title"])
    
    return emergence_order

# merges timesteps that don't have <= k author papers with the previous timestep.
def merge_emergence_order_df(emergence_order: dict, timestep_map: dict) -> dict:
    for timestep in timestep_map:
        if timestep_map[timestep] == timestep:
            continue



def get_all_vecs_df(vecs_and_author_info: pd.DataFrame) -> tuple:
    rev_order = []
    rev_order_title = []

    for row in vecs_and_author_info.iterrows():
        rev_order.append(tuple(ast.literal_eval(row[1]["embedding"])))
        rev_order_title.append(row[1]["title"])
    
    return rev_order, rev_order_title

    
def main(domain: str, field: str, K: int):
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

    if domain == "nobel":
        vecs_path = f"data/nobel_winners/{field}/sbert-labels-and-authors-ordered"
        individual_path = f"data/nobel_winners/{field}/authorship-pickled"
        out_path = f"results/summary/authorship-{field}-{K}.p" 

    for i, filename in enumerate(os.listdir(vecs_path)):
        if filename.endswith(".csv"):
            print(filename)
        vecs_filename = os.path.join(vecs_path, filename)
        
        if os.path.exists(f"{individual_path}/{filename[:-4]}-{K}.p"):
            continue

        vecs_and_authors = pd.read_csv(vecs_filename, names=["timestep", "title", "num_authors", "first_author", "embedding"])
        vecs_and_authors_k = filter_k_authors_df(K, vecs_and_authors)
        emergence_order = get_emergence_order_df(vecs_and_authors)
        all_vecs, all_titles = get_all_vecs_df(vecs_and_authors_k)
        
        if len(all_vecs) < 5:
            continue
        
        name_to_model = {
            "1NN": rank_on_1NN,
            "2NN": make_rank_on_knn(2),
            "3NN": make_rank_on_knn(3),
            "4NN": make_rank_on_knn(4),
            "5NN": make_rank_on_knn(5),
            "Prototype": rank_on_prototype,
            "Progenitor": rank_on_progenitor(all_vecs[0]),
            #"Exemplar": make_rank_on_exemplar(s),
            "Local": rank_on_1NN
        }
        model_types = {
            "1NN": "global",
            "2NN": "global",
            "3NN": "global",
            "4NN": "global",
            "5NN": "global",
            "Prototype": "global",
            "Progenitor": "global",
            "Local": "local"
        }

        res = {}
        ll_rand = get_random_score_k_authors(K, vecs_and_authors)
        log_Ls, model_order = get_probability_score_k_authors(K, vecs_and_authors, emergence_order, all_vecs, all_titles, name_to_model, model_types)
        for name, ll in zip(model_order, log_Ls):
            print(" === MODEL ===")
            print(f"{name}: {ll}")
            diff = ll - ll_rand
            res[name] = (diff, len(all_vecs))
            models[name][filename[:-4]] = (diff, len(all_vecs))
        print("RANDOM:", ll_rand)

        with open(f"{individual_path}/{filename[:-4]}-{K}.p", "wb") as p_file:
            pickle.dump(res, p_file)
    
    with open(out_path, "wb") as out_f:
        pickle.dump(models, out_f)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("-k", help="threshold for number of authors to include (<=k). Input -1 for first author papers only.", type=int, required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "chemistry", "medicine", "economics", "cogsci"])
    args = parser.parse_args()

    main(args.type, args.field, args.k)