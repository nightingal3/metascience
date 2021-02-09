import matplotlib.pyplot as plt
import seaborn
from typing import List
import os
import pandas as pd
import numpy as np

from src.models.predict import get_attested_order, get_emergence_order, rank_on_1NN, get_sim

def get_sim_over_time_multi(names: List, filepath: str) -> List:
    all_sim = []
    for filename in os.listdir(filepath):
        if filename[:-4] in names:
            emergence_order = get_emergence_order(os.path.join(filepath, filename))
            all_vecs = get_attested_order(os.path.join(filepath, filename))

            sim_over_t = get_sim_over_time(emergence_order, all_vecs)
            all_sim.extend([(item[0], item[1], filename[:-4]) for item in sim_over_t])

    return all_sim


def get_sim_over_time(emergence_order: dict, all_vecs: List) -> List:
    last_timestep = max(emergence_order.keys())
    emerged_papers = []
    sim_over_time = []
    first_papers = [p for p in emergence_order[0]]

    for t in range(last_timestep):
        #print("TIMESTEP: ", t)
        curr_papers = emergence_order[t]
        print("curr papers: ", len(curr_papers))
        next_papers = emergence_order[t + 1]
       
        """emerged_papers.extend(curr_papers)
        last_papers = emerged_papers
        available_papers = all_vecs[:]
        for paper in emerged_papers:
            available_papers.remove(
                list(paper)
            )  # Not using list comprehension so duplicates are preserved

        #pdb.set_trace()
        pred_and_sim = rank_on_1NN(emerged_papers, available_papers, keep_sim=True)
        pred, sim = zip(*pred_and_sim)

       
        for vec in next_papers:
            found_index = pred.index(vec)
            sim_over_time.append((t + 1, sim[found_index]))"""
        for vec in next_papers:
            avg = 0
            for p in first_papers:
                avg += get_sim(vec, p)[0][0]
            sim_over_time.append((t + 1, avg))

    
    return sim_over_time

def plot_similarity_over_time(sim_over_time: List, filename: str) -> None:
    time, sim, names = zip(*sim_over_time)
    names = [name.replace("=", " ").replace("_", " ").replace("-", " ") for name in names]
    #seaborn.scatterplot(time, sim, hue=names)
    data = pd.DataFrame(data=np.column_stack((time, sim, names)), columns=["timestep", "sim", "name"])
    fg = seaborn.FacetGrid(data.astype({"timestep": int, "sim": float, "name": str}), hue="name")
    fg.map(seaborn.regplot, "timestep", "sim")
    plt.ylabel("Similarity of emerging papers\nto $S_{0}$", fontsize=30)
    plt.xlabel("Timestep $t$", fontsize=30)
    plt.xticks([i for i in range(5, max(time), 5)], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")  
    plt.show()
    

if __name__ == "__main__":
    best_names = ["Leslie-Lamport", "Donald_E=-Knuth", "Robert_W=-Floyd", "Silvio-Micali", "John_E=-Hopcroft"]
    best_pairwise = ["Geoffrey_E=-Hinton", "Leslie-Lamport", "David_A=-Patterson", "Richard_M=-Karp", "Adi-Shamir"]
    worst_names = ["Adi-Shamir", "Amir-Pnueli", "Tony-Hoare", "Joseph-Sifakis", "David_A=-Patterson"]
    worst_pairwise = ["Juris-Hartmanis", "Robert_E=-Kahn", "John-McCarthy", "Edgar_F=-Codd", "John-Cocke"]
    random_names = ["Donald_E=-Knuth", "Amir-Pnueli", "Yoshua-Bengio", "Ken-Thompson", "John_W=-Backus"]

    emergence_order = get_emergence_order("data/turing_winners/vecs-abstracts-ordered/Juris-Hartmanis.csv")
    all_vecs = get_attested_order("data/turing_winners/vecs-abstracts-ordered/Juris-Hartmanis.csv")
    #sim_over_t = get_sim_over_time(emergence_order, all_vecs)
    #plot_similarity_over_time(sim_over_t)
    sim_over_t = get_sim_over_time_multi(best_pairwise, "data/turing_winners/vecs-abstracts-ordered")
    plot_similarity_over_time(sim_over_t, filename="best-p-scientists-sim")


