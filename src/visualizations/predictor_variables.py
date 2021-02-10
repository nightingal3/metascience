import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from numpy import std
import pickle
import pdb
import os
from statistics import median
from typing import List

# Descriptive stats - number of papers, median or mean number of coauthors, age of oldest paper

def plot_number_papers_domain(domain_filepath: str, fig_name: str, mode: str = "Mean") -> tuple:
    with open(domain_filepath, "rb") as domain_f:
        data = pickle.load(domain_f)
        first_key = list(data.keys())[0]

        num_papers = []
        for scientist in data[first_key]:
            num_papers.append(data[first_key][scientist][1])

        plt.hist(num_papers)
        plt.xlabel("Number of papers published per scientist", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.savefig(f"{fig_name}.png")
        plt.savefig(f"{fig_name}.eps")

        if mode == "Mean":
            return sum(num_papers)/len(num_papers), std(num_papers)
        if mode == "Median":
            return median(num_papers), std(num_papers)


def plot_avg_number_coauthors_domain(domain_dir: str, fig_name: str, field: str, mode: str = "Mean") -> float:
    num_coauthors = []
    for filename in os.listdir(domain_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(domain_dir, filename)
            
            if field == "cs":
                df = pd.read_csv(full_path, names=["year", "title", "num_authors", "first_author"])
            else:
                df = pd.read_csv(full_path, names=["title", "num_authors", "first_author"])
            if mode == "Mean":
                num_coauthors.append(df["num_authors"].mean())
            elif mode == "Median":
                num_coauthors.append(df["num_authors"].median())
    
    plt.hist(num_coauthors)
    plt.xlabel(f"{mode} number of coauthors on papers", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.savefig(f"{fig_name}.png")
    plt.savefig(f"{fig_name}.eps")

    return sum(num_coauthors)/len(num_coauthors), std(num_coauthors)
    

def plot_age_papers_domain(domain_dir: str, fig_name: str, field: str, mode: str = "Oldest") -> List:
    paper_years = []
    oldest_or_med = []

    for filename in os.listdir(domain_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(domain_dir, filename)

            df = pd.read_csv(full_path, usecols=[0], names=["year"])
            if len(df["year"]) == 0:
                continue
            paper_years.extend(list(df["year"]))

            if mode == "Oldest":
                oldest_or_med.append(df["year"].min())
            elif mode == "Median":
                oldest_or_med.append(df["year"].median())
    plt.hist(paper_years)
    plt.xlabel("Year of publication", fontsize=14)
    plt.ylabel("Number of papers", fontsize=14)
    plt.savefig(f"{fig_name}.png")
    plt.savefig(f"{fig_name}.eps")

    return oldest_or_med
    

if __name__ == "__main__":
    field = "physics"
    mode = "Median"
    stat = "year"

    if stat == "coauthors":
        if field == "cs":
            authorship_dir = "data/turing_winners/authorship"
        else:
            authorship_dir = f"data/nobel_winners/{field}/authorship"
        print(plot_avg_number_coauthors_domain(authorship_dir, f"{field}-coauthors-{mode}", field, mode=mode))

    elif stat == "num_papers":
        print(plot_number_papers_domain(f"results/full-2/{field}.p", f"{field}-num_papers-{mode}", mode=mode))
    
    elif stat == "year":
        if field == "cs":
            field_dir = "data/turing_winners/sbert-abstracts"
        else:
            field_dir = f"data/nobel_winners/{field}/sbert-abstracts"

        oldest = plot_age_papers_domain(field_dir, f"year-{field}", field)
        print(oldest)
        print(median(oldest), std(oldest))