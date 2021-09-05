import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, kruskal

import argparse
import os
import pickle
import statistics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="run comparison (across all fields)", action="store_true")
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci)", choices=["cs", "physics", "chemistry", "medicine", "economics", "cogsci"])
    parser.add_argument("--model", help="which model type to optimize", choices=["exemplar", "kNN", "progenitor", "prototype", "local"])
    args = parser.parse_args()

    field, do_comparison, model_type = args.field, args.c, args.model
    fields = ["cs", "physics", "chemistry", "medicine", "economics"]

    if do_comparison:
        comparison = []
        for field in fields:
            if field == "cs":
                s_vals_path = f"data/turing_winners/individual-s-vals-{model_type}/"
            else:
                s_vals_path = f"data/nobel_winners/{field}/individual-s-vals-{model_type}"

            s_vals = []
            for filename in os.listdir(s_vals_path):
                if not filename.endswith(".p"):
                    continue
                filename_s = os.path.join(s_vals_path, filename)
                try:
                    s_val_dict = pickle.load(open(filename_s, "rb"))
                except:
                    print(f"Failed on {filename}")
                s_vals.append(s_val_dict["s"])
            comparison.append(s_vals)

        print(kruskal(*comparison))

    else:
        if field == "cs":
            s_vals_path = f"data/turing_winners/individual-s-vals-{model_type}"
        else:
            s_vals_path = f"data/nobel_winners/{field}/individual-s-vals-{model_type}"

        s_vals = []
        for filename in os.listdir(s_vals_path):
            if not filename.endswith(".p"):
                continue
            filename_s = os.path.join(s_vals_path, filename)
            try:
                s_val_dict = pickle.load(open(filename_s, "rb"))
            except:
                print(f"Failed on {filename}")
            s_vals.append(s_val_dict["s"])

        print(statistics.mean(s_vals))
        print(statistics.median(s_vals))
        print(statistics.stdev(s_vals))

        plt.hist(s_vals, density=False, weights=np.ones(len(s_vals))/len(s_vals))
        plt.title(f"Distribution of exemplar parameter in {field} field")
        plt.ylabel("Density")
        plt.xlabel("Exemplar parameter")
        plt.savefig(f"dist_s_vals-{field}.png")
        plt.savefig(f"dist_s_vals-{field}.eps")




        