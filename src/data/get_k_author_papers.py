import pandas as pd
import os

def get_first_author_papers(authors_filename: str, vectors_filename: str, out_filename: str) -> None:
    authors = pd.read_csv(authors_filename, header=None)
    vecs = pd.read_csv(vectors_filename, header=None)
    assert authors.shape[0] == vecs.shape[0]
    first_author_rows = authors[3] == 1
    first_author_vecs = vecs[first_author_rows == True]

    first_author_vecs.to_csv(out_filename, header=False, index=False)

def get_k_author_papers(k: int, authors_filename, vectors_filename, out_filename: str) -> None:
    authors = pd.read_csv(authors_filename, header=None)
    vecs = pd.read_csv(vectors_filename, header=None)
    assert authors.shape[0] == vecs.shape[0]
    k_author_rows = authors[2] <= k
    k_author_vecs = vecs[k_author_rows == True]

    k_author_vecs.to_csv(out_filename, header=False, index=False)

def get_k_author_papers_same_fi
le(k: int, filename: str, out_filename: str) -> None:
    df = pd.read_csv()

def get_all_first_author_papers_dir(authors_dir: str, vecs_dir: str, out_dir: str) -> None:
    for filename in os.listdir(authors_dir):
        if filename.endswith(".csv"):
            print(filename)
            authors = os.path.join(authors_dir, filename)
            vecs = os.path.join(vecs_dir, filename)
            out = os.path.join(out_dir, filename)

            get_first_author_papers(authors, vecs, out)

def get_all_k_authors_in_dir(k: int, authors_dir: str, vecs_dir: str, out_dir: str) -> None:
    for filename in os.listdir(authors_dir):
        if filename.endswith(".csv"):
            print(filename)
            authors = os.path.join(authors_dir, filename)
            vecs = os.path.join(vecs_dir, filename)
            out = os.path.join(out_dir, filename)

            get_k_author_papers(k, authors, vecs, out)

if __name__ == "__main__":
    #get_k_author_papers(3, "data/turing_winners/vecs-abstracts-w-labels/labels-and-authors/Adi-Shamir.csv", "data/turing_winners/vecs-abstracts-w-labels/abstract-vectors/Adi-Shamir.csv", "data/turing_winners/three-authors-only/Adi-Shamir.csv")
    #get_all_first_author_papers_dir("data/turing_winners/vecs-abstracts-w-labels/sbert-labels-and-authors", "data/turing_winners/sbert-abstracts", "data/turing_winners/sbert-first-author-only")
    get_all_k_authors_in_dir(3, "data/turing_winners/vecs-abstracts-w-labels/sbert-labels-and-authors", "data/turing_winners/sbert-abstracts", "data/turing_winners/sbert-three-authors-only")