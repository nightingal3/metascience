import csv
import os

from pandas import DataFrame, read_csv
import pdb

def rank_by_date(in_filename: str, out_filename: str = "../../data/ordered_by_date.csv") -> None:
    try:
        print(in_filename)
        df = read_csv(in_filename, header=None)
        all_years = sorted(df.groupby(df.columns[0]).groups.keys())
        timesteps = {all_years[i]: i for i in range(len(all_years))}
        df = df.replace(timesteps)
        #if len(df.columns) > 2:
            #df = df.truncate(after=1, axis="columns")
        df.to_csv(out_filename, index=False, header=False)

    except IOError:
        print("Error in file I/O, check filenames")


def emergence_order_all_in_dir(in_dir: str, out_dir: str) -> None:
    for filename in os.listdir(in_dir):
        print(filename)
        if filename.endswith(".csv"):
            try:
                rank_by_date(in_dir + filename, out_dir + filename)
            except:
                continue

if __name__ == "__main__":
    rank_by_date("./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final-ordered.csv")
    #rank_by_date("./data/turing_winners/abstracts/Geoff-Hinton-abstract-vecs.csv", "./data/turing_winners/abstracts/Geoff-Hinton-abstract-ordered.csv")
    assert False
    emergence_order_all_in_dir("data/turing_winners/vecs-abstracts-only/", "data/turing_winners/vecs-abstracts-ordered/")

