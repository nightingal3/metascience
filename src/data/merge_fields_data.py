from math import floor
import os
import pandas as pd
import pickle
import pdb

# 1NN LL advantage vs random ~ 1 + num_papers + num_collaborators + oldest_paper + (1 + num_papers | field) + (1 + num_collaborators | field)

fields = ["cs", "chemistry", "economics", "medicine", "physics"]

def aggregate(out_filename: str) -> None:
    data = []
    for field in fields:
        ll_path = f"results/full-2/{field}.p"
        if field == "cs":
            authorship_path = "data/turing_winners/authorship"
            year_path = "data/turing_winners/sbert-abstracts"
        else:
            authorship_path = f"data/nobel_winners/{field}/authorship"
            year_path = f"data/nobel_winners/{field}/sbert-abstracts"
        
        scientist_data = pickle.load(open(ll_path, "rb"))
        for scientist in scientist_data["1NN"].keys():
            ll_diff_1NN = scientist_data["1NN"][scientist][0]
            ll_diff_prototype = scientist_data["Prototype"][scientist][0]
            ll_diff_exemplar = scientist_data["Exemplar"][scientist][0]

            vecs_path = os.path.join(year_path, f"{scientist}.csv") 
            if not os.path.exists(vecs_path):
                continue

            df_year = pd.read_csv(vecs_path, usecols=[0], names=["year"])
            df_year["year"] = df_year["year"].dropna()
            num_papers = len(df_year)
            if num_papers < 5:
                continue
            oldest_paper_age = df_year["year"].min()
            decade_group = floor(oldest_paper_age / 10) * 10

            scientist_authorship_path = os.path.join(authorship_path, f"{scientist}.csv")
            if not os.path.exists(scientist_authorship_path):
                continue

            if field == "cs":
                df_collab = pd.read_csv(scientist_authorship_path, names=["year", "title", "num_authors", "first_author"])
            else:
                df_collab = pd.read_csv(scientist_authorship_path, names=["title", "num_authors", "first_author"])

            median_num_collaborators = df_collab["num_authors"].median()

            data.append([scientist, ll_diff_1NN, ll_diff_prototype, ll_diff_exemplar, num_papers, median_num_collaborators, oldest_paper_age, decade_group, field])

    data_df = pd.DataFrame(data, columns=["scientist_name", "ll_diff_1NN", "ll_diff_proto", "ll_diff_exemplar", "num_papers", "median_collaborators", "oldest_paper", "decade_bin", "field"])
    data_df.to_csv(out_filename, index=False)

if __name__ == "__main__":
    aggregate("data/merged_data.csv")