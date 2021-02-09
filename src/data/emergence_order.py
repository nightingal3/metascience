import csv
import os

from pandas import DataFrame, read_csv
import pdb

def rank_by_date(in_filename: str, out_filename: str = "../../data/ordered_by_date.csv", sort_by_date: bool = False) -> None:
    try:
        df = read_csv(in_filename, header=None)
        if sort_by_date:
            df = df.sort_values(by=0, ascending=False)
        all_years = sorted(df.groupby(df.columns[0]).groups.keys())
        timesteps = {all_years[i]: i for i in range(len(all_years))}
        df = df.replace(timesteps)
        #if len(df.columns) > 2:
            #df = df.truncate(after=1, axis="columns")
        df.to_csv(out_filename, index=False, header=False)

    except IOError:
        print("Error in file I/O, check filenames")


def emergence_order_all_in_dir(in_dir: str, out_dir: str, sort_by_date: bool = False) -> None:
    for filename in os.listdir(in_dir):
        print(filename)
        if filename.endswith(".csv"):
            try:
                rank_by_date(in_dir + filename, out_dir + filename, sort_by_date=sort_by_date)
            except:
                continue

if __name__ == "__main__":
    #rank_by_date("./data/turing_winners/abstracts/geoff/Geoff-Hinton-sbert.csv", "./data/turing_winners/first-author-ordered/Geoffrey_E=-Hinton.csv")
    #rank_by_date("./data/turing_winners/vecs-abstracts-ordered/Geoffrey_E=-Hinton.csv", "./data/turing_winners/vecs-abstracts-ordered/Geoffrey_E=-Hinton-ordered.csv")
    #assert False
    # with open("data/nobel_winners/physics/physics.csv") as winner_f:
    #     winner_ids = []
    #     reader = csv.reader(winner_f)
    #     for line in reader:
    #         scientist_id = line[1]
    #         if scientist_id != "N/A" and scientist_id != "None":
    #             winner_ids.append(scientist_id)
    # for filename in os.listdir("data/nobel_winners/physics/random-sample/sbert-abstracts/"):
    #     if filename[:-4] in winner_ids:
    #         print(filename)
    # assert False
    emergence_order_all_in_dir("data/nobel_winners/chemistry/sbert-labels-and-authors/", "data/nobel_winners/chemistry/sbert-labels-and-authors-ordered/", sort_by_date=True)

