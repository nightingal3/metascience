import csv
import os

from pandas import DataFrame, read_csv


def rank_by_date(in_filename: str, out_filename: str = "../../data/ordered_by_date.csv") -> None:
    try:
        print(in_filename)
        df = read_csv(in_filename, header=None)
        all_years = sorted(df.groupby(df.columns[0]).groups.keys())
        timesteps = {all_years[i]: i for i in range(len(all_years))}
        df = df.replace(timesteps)
        df.to_csv(out_filename, index=False, header=False)

    except IOError:
        print("Error in file I/O, check filenames")


def emergence_order_all_in_dir(in_dir: str, out_dir: str) -> None:
    for filename in os.listdir(in_dir):
        print(filename)
        if filename.endswith(".csv"):
            rank_by_date(in_dir + filename, out_dir + filename)

if __name__ == "__main__":
    emergence_order_all_in_dir("data/turing_winners/vecs/", "data/turing_winners/ordered/")

