import argparse
import os
from pprint import pprint
from random import shuffle
from typing import List

import pandas as pd

field_scientists = {
    "cs": 69,
    "chemistry": 120,
    "economics": 74,
    "medicine": 151,
    "physics": 168
}
def filter_prominent_scientists(scientist_ids_filepath: str, random_scientists_dir: str) -> List:
    with open(scientist_ids_filepath, "r") as ids_f:
        prominent_scientists = ids_f.readlines()

    removed = []
    for filename in os.listdir(random_scientists_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(random_scientists_dir, filename)
            if filename[:-4] in prominent_scientists:
                removed.append(filename)
                os.remove(full_path)

    return removed


def filter_few_papers(random_scientists_dir: str) -> List:
    removed = []
    for filename in os.listdir(random_scientists_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(random_scientists_dir, filename)
            num_lines = sum(1 for line in open(full_path))

            if num_lines < 5:
                os.remove(full_path)
                removed.append(filename)

    return removed

def select_same_size_subset(field: str, random_scientists_dir: str) -> List:
    all_random_scientists = list(os.listdir(random_scientists_dir))
    shuffle(all_random_scientists)

    keep, removed = all_random_scientists[:field_scientists[field]], all_random_scientists[field_scientists[field]:]

    for filename in os.listdir(random_scientists_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(random_scientists_dir, filename)
            if filename in removed:
                os.remove(full_path)

    return removed
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="'turing' for turing winners and 'nobel' for others", choices=["turing", "nobel"], required=True)
    parser.add_argument("--field", help="which field to process (physics/chem/medicine/econ/cogsci", choices=["physics", "chemistry", "medicine", "economics", "cogsci"])
    args = parser.parse_args()

    if args.type == "nobel":
        base_path = "data/nobel_winners/"
    else:
        base_path = "data/turing_winners/"

    removed_path = f"{base_path}{args.field}/random-sample/removed_random.txt"

    if args.field:
        removed = filter_prominent_scientists(f"{base_path}{args.field}/{args.field}_winner_ids.csv", f"{base_path}{args.field}/random-sample/sbert-abstracts-ordered/")
        removed.extend(select_same_size_subset(args.field, f"{base_path}{args.field}/random-sample/sbert-abstracts-ordered/"))
    else:
        removed = select_same_size_subset("cs", f"{base_path}random-sample/sbert-abstracts-ordered")
        removed_path = f"{base_path}random-sample/removed_random.txt"

    pprint(removed)
    with open(removed_path, "w") as out_f:
        for item in removed:
            out_f.write(f"{item}\n")