import csv

from pandas import DataFrame, read_csv


def rank_by_date(in_filename: str, out_filename: str = "../../data/ordered_by_date.csv") -> None:
    try:
        df = read_csv(in_filename)
        all_years = sorted(df.groupby(df.columns[0]).groups.keys())
        timesteps = {all_years[i]: i for i in range(len(all_years))}
        df = df.replace(timesteps)
        df.to_csv(out_filename, index=False)

    except IOError:
        print("Error in file I/O, check filenames")


if __name__ == "__main__":
    rank_by_date("../../data/hinton_paper_vectors.csv")

