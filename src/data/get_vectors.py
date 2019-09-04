import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ast
import csv
from functools import reduce
import io
from typing import Dict

import nltk
nltk.download("stopwords")

punctuation = [".", ",", "'", "\"", ":", ";", "?", "(", ")", "[", "]"]


def preproc_pubs(filename: str, stop_words: list, lemmatizer: WordNetLemmatizer, out_filename: str) -> None:
    cleaned_rows = []
    with open(filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            year, title_ch = row
            title_words = (
                "".join(
                    ch for ch in title_ch if ch not in punctuation)).lower().split(" ")
            clean_title = " ".join([lemmatizer.lemmatize(word)
                                    for word in title_words if word not in stop_words])
            cleaned_rows.append([year, clean_title])
    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        for row in cleaned_rows:
            writer.writerow(row)


def load_vectors(fname: str) -> Dict:
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def make_pubs_vectors(in_filename: str, vectors: Dict, out_filename: str) -> None:
    vec_rows = []
    with open(in_filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            year, title = row
            title = title.split(" ")
            sum_vec = np.zeros(300)
            valid_words = 0
            for word in title:
                if word in vectors:
                    vec = vectors[word]
                    sum_vec += vec
                    valid_words += 1
            avg_vec = [float(val) / valid_words for val in sum_vec]
            print(avg_vec)
            vec_rows.append([year, avg_vec])

    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        for row in vec_rows:
            writer.writerow(row)


def make_tsv_files(in_metadata_filename: str, in_vecs_filename: str, out_vecs_filename: str, out_metadata_filename: str) -> None:
    metadata_rows = []
    vec_rows = []
    with open(in_metadata_filename, "r") as metadata_f, open(in_vecs_filename, "r") as vecs_f:
        reader_m = csv.reader(metadata_f)
        reader_v = csv.reader(vecs_f)

        for metadata, vec in zip(reader_m, reader_v):
            metadata_rows.append(metadata)
            vec_rows.append(ast.literal_eval(vec[1]))

    with open(out_metadata_filename, "w") as out_metadata, open(out_vecs_filename, "w") as out_vecs:
        writer_m = csv.writer(out_metadata, delimiter="\t")
        writer_v = csv.writer(out_vecs, delimiter="\t")
        writer_m.writerow(["Year", "Title"])

        for metadata_row, vec_row in zip(metadata_rows, vec_rows):
            writer_m.writerow(metadata_row)
            writer_v.writerow(vec_row)


if __name__ == "__main__":
    vecs = load_vectors("./data/wiki-news-300d-1M.vec")
    """make_pubs_vectors(
        "./data/hinton_papers_cleaned.csv",
        vecs,
        "./data/hinton_paper_vectors.csv")"""
    make_tsv_files(
        "./data/hinton_titles_selected.csv",
        "./data/hinton_vecs_selected.csv",
        "./data/hinton_vecs_selected.tsv",
        "./data/hinton_titles_selected.tsv")
