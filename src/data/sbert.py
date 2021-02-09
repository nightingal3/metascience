import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
import pdb

def make_pubs_vectors(in_filename: str, model: SentenceTransformer, out_filename: str, has_abstract: bool = False, by_type=False, abstract_only: bool = False, abs_by_sent: bool = False) -> None:
    vec_rows = []
    with open(in_filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if has_abstract:
                try:
                    year, title, abstract = row
                    if abstract == "n/a" and abstract_only:
                        continue
                    if abs_by_sent:
                        abstract_sents = list(filter(lambda x: x != "", abstract.split(".")))
                    else:
                        abstract = [abstract]

                except ValueError:
                    year, title = row
                    abstract = "n/a"
                #abstract = abstract.split(" ")
                if by_type:
                    abstract = set(abstract)
            else:
                year, title = row
            #title = title.split(" ")
            title = [title]
            if by_type:
                title = set(title)
            
            title_vec = list(model.encode(title)[0])
        
            if has_abstract:
                if abs_by_sent:
                    abstract_vec = model.encode([abstract_sents[0]])[0]
                    for sent in abstract_sents[1:]:
                        abstract_vec = abstract_vec + model.encode([sent])[0]
                    abstract_vec = abstract_vec / len(abstract_sents)

                else:
                    abstract_vec = list(model.encode(abstract)[0])
                  
            if has_abstract and not abstract_only:
                #vec_rows.append([year, title_vec, abstract_vec])
                vec_rows.append([year, " ".join(title), title_vec])
            elif has_abstract and abstract_only:
                vec_rows.append([year, '"' + " ".join(title) + '"', abstract_vec])
            else:
                vec_rows.append([year, " ".join(title), title_vec])

    np.savetxt(out_filename, vec_rows, delimiter=",", fmt="%s")


def make_pubs_vectors_in_dir(dirname: str, out_dir: str, model: SentenceTransformer, has_abstract: bool = False, abstract_only: bool = False, abs_by_sent: bool = False) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            make_pubs_vectors(full_path, model, new_path, has_abstract=has_abstract, abstract_only=abstract_only, abs_by_sent=abs_by_sent)


if __name__ == "__main__":
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    #make_pubs_vectors("./data/turing_winners/abstracts/geoff/Geoff-Hinton-cleaned.csv", model, "./data/turing_winners/abstracts/geoff/Geoff-Hinton-sbert.csv", has_abstract=True, abstract_only=True, abs_by_sent=False)
    make_pubs_vectors_in_dir("./data/nobel_winners/medicine/random-sample/abstracts-cleaned", "./data/nobel_winners/medicine/random-sample/sbert-abstracts", model, has_abstract=True, abstract_only=True, abs_by_sent=False)