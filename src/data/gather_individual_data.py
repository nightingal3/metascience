import argparse
import os
import pickle

if __name__ == "__main__":
    in_dir = "data/pickled/abstracts/worst-fixed/"
    models = {"1NN": {}, "2NN": {}, "3NN": {}, "4NN": {}, "5NN": {}, "Prototype": {}, "Progenitor": {}, "Exemplar": {}, "Local": {}}
    for filename in os.listdir(in_dir):
        print(filename)
        if filename[-2:] == ".p":
            individual_name = filename[:-12]
            individual_data = pickle.load(open(in_dir + filename, "rb"))
            for model in models:
                models[model][individual_name] = (individual_data[model][0], individual_data[model][2])
    print(models["1NN"])
    pickle.dump(models, open("data/pickled/abstracts/models-worst.p", "wb"))
