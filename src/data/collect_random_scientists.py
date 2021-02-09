import json
import os
from pprint import pprint
import random
from typing import List
import pdb

def filter_work(paper_data: dict) -> bool:
    is_valid = (paper_data["id"] != "" and paper_data["title"] != "") and (len(paper_data["authors"]) != 0 and len(paper_data["fieldsOfStudy"]) != 0)

    return is_valid
        
def collect_field_of_study(all_data: List) -> tuple:
    fields = set()
    authors_by_field = {}
    for data in all_data:
        all_fields = data["fieldsOfStudy"]
        fields.update(all_fields)
        if all_fields[0] not in authors_by_field:
            authors_by_field[all_fields[0]] = set()
        try:
            authors_by_field[all_fields[0]].update([info["ids"][0] for info in data["authors"]])
        except:
            continue

    return fields, authors_by_field

def random_out(data: List, filename: str, k: int = 100) -> None:
    data = random.sample(data, k)

    with open(filename, "w") as out_f:
        out_f.writelines(s + '\n' for s in data)


if __name__ == "__main__":
    all_data = []

    for f in os.listdir("data/external/semanticscholar"):
        with open(f"data/external/semanticscholar/{f}", "r") as json_f:
            data = [json.loads(line) for line in json_f]
            all_data.extend(data)
    
    all_data = list(filter(filter_work, all_data))
    print(len(all_data))
    _, authors = collect_field_of_study(all_data)

    random_out(list(authors["Physics"]), "data/nobel_winners/random_physicists_2.txt", k=300)
    #random_out(list(authors["Chemistry"]), "data/nobel_winners/random_chemists.txt", k=300)
    #random_out(list(authors["Computer Science"]), "data/turing_winners/random_cs.txt", k=300)
    #random_out(list(authors["Economics"]), "data/nobel_winners/random_economists.txt", k=300)
    #random_out(list(authors["Physics"]), "data/nobel_winners/random_physicists.txt", k=300)
    #random_out(list(authors["Physics"]), "data/nobel_winners/random_physicists.txt", k=300)






