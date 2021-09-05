import os
import pickle
import pdb
from random import shuffle

scientist_sample_size = {
    "cs": 69, 
    "chemistry": 120,
    "economics": 74,
    "medicine": 151,
    "physics": 168
}

def match_sample_size(field: str, results_path: str) -> dict:
    models = pickle.load(open(results_path, "rb"))
    scientist_names = list(models["kNN"].keys())
    shuffle(scientist_names)

    keep, _ = scientist_names[:scientist_sample_size[field]], scientist_names[scientist_sample_size[field]:]

    new_models = {model_name: {} for model_name in models}
    for model_name in models:
        for scientist in models[model_name]:
            if scientist in keep:
                new_models[model_name][scientist] = models[model_name][scientist]

    return new_models

def match_saved_scientists(fixed_filepath: str, in_filepath: str) -> None:
    models_match = pickle.load(open(fixed_filepath, "rb"))
    selected_scientists = list(models_match['kNN'].keys())

    models_original = pickle.load(open(in_filepath, "rb"))
    models_fixed = {model_name: {} for model_name in models_original}

    for model_name in models_original:
        for scientist_name in models_original[model_name]:
            if scientist_name in selected_scientists:
                models_fixed[model_name][scientist_name] = models_original[model_name][scientist_name]
    
    return models_fixed

def match_saved_scientists_in_dir(ref_filepath: str, in_dir: str, out_dir: str) -> None:
    for filename in os.listdir(in_dir):
        if not filename.endswith(".p"):
            continue
        in_path = os.path.join(in_dir, filename)
        out_path = os.path.join(out_dir, filename)
        models_fixed = match_saved_scientists(ref_filepath, in_path)
        
        with open(out_path, "wb") as out_f:
            pickle.dump(models_fixed, out_f)

if __name__ == "__main__":
    #path = "./results/full-new/physics-random.p"
    #new_models = match_sample_size("physics", path)

    #with open("./results/full-fixed/physics-random.p", "wb") as out_f:
        #pickle.dump(new_models, out_f)
    match_saved_scientists_in_dir("./results/full-fixed/medicine-random.p", "./results/shuffle-ll-new-random-sample/medicine/", "./results/shuffle-ll-new-random-sample-fixed/medicine/")
    #match_saved_scientists("./results/full-fixed/cs-random.p", "./results/summary/k-author/authorship-cs-random-1.p", "./results/summary/k-author-fixed/authorship-cs-random-1.p")