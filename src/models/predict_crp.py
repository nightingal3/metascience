import argparse
import os
import pickle
import pdb

from chinese_restaurant import CRP
from predict import get_attested_order, get_emergence_order, get_probability_rand, get_probability_score_crp

def main(field: str):
    if field == "cs":
        vecs_path = f"data/turing_winners/sbert-abstracts-ordered"
        order_path = vecs_path
    else:
        vecs_path = f"data/nobel_winners/{field}/abstracts-ordered"
        order_path = vecs_path
    
    out_path = f"results/summary/full/{field}-crp.p"

    models = {
        "prototype-crp": {},
        "progenitor-crp": {},
        "exemplar-crp": {},
        "local-crp": {},
        "Null": {}
    }
    
    if field == "cs":
        individual_s_val_path = "data/turing_winners/individual-s-vals"
        individual_out_path = "data/turing_winners/pickled-full-crp"
    else:
        individual_s_val_path = f"data/nobel_winners/{field}/individual-s-vals"
        individual_out_path = f"data/nobel_winners/{field}/pickled-full-crp"
        

    if "random" in field:
        vecs_path = f"data/nobel_winners/{field}/sbert-abstracts-ordered"
        order_path = vecs_path
    
    for i, filename in enumerate(os.listdir(f"{individual_s_val_path}/exemplar-crp")):
        if filename.endswith(".p"):
            print(filename[:-15])

        res = {}
        
        scientist_name = filename[:-15]
        csv_name = f"{scientist_name}.csv"
        vecs_filename = os.path.join(vecs_path, csv_name)
        order_filename = os.path.join(order_path, csv_name)
        out_filename = os.path.join(individual_out_path, filename)
        
        if not os.path.exists(vecs_filename):
            continue

        all_vecs = get_attested_order(vecs_filename, vecs_col=2, multicols=True)
        emergence_order = get_emergence_order(order_filename, vecs_col=2, multicols=True)

        if len(all_vecs) < 5:
            continue

        # if os.path.exists(out_filename):
        #     with open(out_filename, "rb") as done_f:
        #         done_models = pickle.load(done_f)
        #         for model in done_models:
        #             models[model][scientist_name] = done_models[model]
        #     ll_rand = get_probability_rand(emergence_order)
        #     models["Null"][scientist_name] = ll_rand
        #     print("Already done this scientist")
        #     continue
        model_s_vals = {}
        unreadable = False
        
        for model_name in models:
            if model_name == "Null":
                continue
            scientist_s_val_path = os.path.join(individual_s_val_path, f"{model_name}/{scientist_name}-{model_name}.p")
            
            if not os.path.exists(scientist_s_val_path):
                continue

            with open(scientist_s_val_path, "rb") as s_file:
                try: 
                    vals = pickle.load(s_file)
                    model_s_vals[model_name] = vals
                except:
                    print("Could not read s-val file")
                    unreadable = True

        if unreadable: 
            continue
        
        res = {}

        ll_rand = get_probability_rand(emergence_order)
        models["Null"][scientist_name] = ll_rand
        print("RANDOM SCORE: ", ll_rand)

        for model_name in model_s_vals:
            model = CRP(alpha=model_s_vals[model_name]["alpha"], starting_points=emergence_order[0], likelihood_type=model_name[:-4], likelihood_hyperparam=model_s_vals[model_name]["s"], use_pca=False, clustering_type="kmeans")
            score = get_probability_score_crp(emergence_order, all_vecs, model, return_log_L_only=True, suppress_print=True)
            print("=== MODEL ===")
            print(f"{model_name}: {score}")
            diff = score - ll_rand
            res[model_name] = (diff, len(all_vecs))
            models[model_name][scientist_name] = (diff, len(all_vecs))

        with open(f"{individual_out_path}/{scientist_name}.p", "wb") as p_file:
            pickle.dump(res, p_file)
    
    with open(out_path, "wb") as p_file:
        pickle.dump(models, p_file)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "physics-random", "chemistry", "chemistry-random", "medicine", "medicine-random", "economics", "economics-random", "cs-random"])
    args = parser.parse_args()
    
    field = args.field
    if args.type == "turing":
        field = "cs"

    main(field)
