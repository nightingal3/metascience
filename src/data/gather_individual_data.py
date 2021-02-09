import argparse
import os
import pickle

if __name__ == "__main__":
    in_dir = "data/nobel_winners/medicine/pickled-tuning/"
    #models = {"1NN": {}, "2NN": {}, "3NN": {}, "4NN": {}, "5NN": {}, "Prototype": {}, "Progenitor": {}, "Exemplar (s=1)": {}, "Exemplar": {},  "Local": {}}
    s_vals = [0.01 * i for i in range(1, 101)]
    models = {val: {} for val in s_vals}

    for filename in os.listdir(in_dir):
        print(filename)
        if filename[-2:] == ".p":
            individual_name = filename[:-2]
            print(individual_name)
            individual_data = pickle.load(open(in_dir + filename, "rb"))
            for model in models:
                models[model][individual_name] = individual_data[model]
    #print(models[0.02])
    pickle.dump(models, open("results/summary/medicine-s-vals.p", "wb"))
