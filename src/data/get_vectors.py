import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ast
import csv
from functools import reduce
import io
import os
from typing import Dict
import pdb
import re

import nltk
nltk.download("stopwords")

punctuation = [".", ",", "'", "\"", ":", ";", "?", "(", ")", "[", "]"]

"""
    Abstract cleaning:
    1. Remove bracketed words
    2. Remove latex 
    3. Remove prefix "Abstract/Summary"
    4. Remove numbers
    5. Remove subheadings (background/results/conclusion)
    6. Remove whitespace (newlines)
"""
def clean_abstract(abstract: str) -> str:
    #abstract = abstract.decode('utf-8','ignore').encode("utf-8")
    if abstract[:8].lower() == "abstract":
        abstract = abstract[8:]
    elif abstract[:7].lower() == "highlights":
        abstract = abstract[7:]
    no_brackets = re.sub("([\(\[]).*?([\)\]])", "", abstract)
    no_latex = re.sub("\\\\\w*", "", no_brackets)
    no_digits = re.sub("\d+\.*\d*%*", "", no_latex)
    no_subheadings = re.sub("(background|results|conclusion)\n*", "", no_digits, flags=re.IGNORECASE)
    no_newlines = re.sub("\n|\t", "", no_subheadings)

    return no_newlines


def preproc_pubs(filename: str, stop_words: list, lemmatizer: WordNetLemmatizer, out_filename: str, get_abstract: bool = False) -> None:
    cleaned_rows = []
    titles_seen = set()
    with open(filename, "r", encoding="utf-8", errors="ignore") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if get_abstract:
                year, title_ch, abstract = row
            else:
                year, title_ch = row
            title_words = (
                "".join(
                    ch for ch in title_ch if ch not in punctuation)).lower().split(" ")
            # papers with the same name/content may appear multiple times
            if "".join(title_words) in titles_seen: 
                continue

            clean_title = " ".join([lemmatizer.lemmatize(word)
                                    for word in title_words if word not in stop_words])
            if abstract:
                cleaned_abstract = clean_abstract(abstract)
                abstract_words = "".join(
                    ch for ch in cleaned_abstract if ch not in punctuation).lower().split(" ")
                lemmatized_abstract = " ".join([lemmatizer.lemmatize(word)
                                    for word in abstract_words if word not in stop_words])
            if abstract:
                cleaned_rows.append([year, clean_title, lemmatized_abstract])
            else:
                cleaned_rows.append([year, clean_title])

            titles_seen.add("".join(title_words))

    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        for row in cleaned_rows:
            writer.writerow(row)


def join_cells(filename: str, out_filename: str) -> None:
    joined_rows = []
    with open(filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            joined_abstract = " ".join(row[2:])
            joined_rows.append([row[0], row[1], joined_abstract])

    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        for row in joined_rows:
            writer.writerow(row)

def split_file(filename: str, out_filename_0: str, out_filename_1: str) -> None:
    f_0 = []
    f_1 = []
    with open(filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            f_0.append([row[0], row[1]])
            f_1.append([row[0], row[2]])
    
    with open(out_filename_0, "w") as out_f_0:
        writer = csv.writer(out_f_0)
        for row in f_0:
            writer.writerow(row)
    
    with open(out_filename_1, "w") as out_f_1:
        writer = csv.writer(out_f_1)
        for row in f_1:
            writer.writerow(row)

def load_vectors(fname: str) -> Dict:
    print("loading...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    print("done loading")
    return data


def make_pubs_vectors(in_filename: str, vectors: Dict, out_filename: str, has_abstract: bool = False, by_type=False, abstract_only: bool = False) -> None:
    vec_rows = []
    with open(in_filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if has_abstract:
                try:
                    year, title, abstract = row
                except ValueError:
                    year, title = row
                    abstract = "n/a"
                abstract = abstract.split(" ")
                if by_type:
                    abstract = set(abstract)
            else:
                year, title = row
            title = title.split(" ")
            print(title)
            if by_type:
                title = set(title)
            sum_vec = np.zeros(300)
            sum_vec_abs = np.zeros(300)
            valid_words = 0
            valid_words_abs = 0

            for word in title:
                if word in vectors:
                    vec = np.asarray(list(vectors[word]))      
                    #pdb.set_trace()
                    try:           
                        sum_vec = np.add(sum_vec, vec)
                    except:
                        continue
                    valid_words += 1
            if has_abstract:
                for word in abstract:
                    if word in vectors:
                        if word == "n/a":
                            valid_words_abs = 0
                            break
                        vec = np.asarray(list(vectors[word]))      
                    #pdb.set_trace()
                    try:           
                        sum_vec_abs = np.add(sum_vec, vec)
                    except:
                        continue
                    valid_words_abs += 1


            if (valid_words > 0 and (valid_words_abs > 0 and has_abstract)) or (abstract_only and valid_words_abs > 0):
                if not abstract_only:
                    avg_vec = [float(val) / valid_words for val in sum_vec]
                if has_abstract:
                    avg_vec_abs = [float(val) / valid_words_abs for val in sum_vec_abs]
                if has_abstract and not abstract_only:
                    vec_rows.append([year, avg_vec, avg_vec_abs])
                elif has_abstract and abstract_only:
                    vec_rows.append([year, " ".join(title), avg_vec_abs])
                else:
                    vec_rows.append([year, avg_vec, " ".join(title)])
            else:
                continue

    with open(out_filename, "w+") as out_file:
        writer = csv.writer(out_file)
        for row in vec_rows:
            writer.writerow(row)


def make_tsv_files(in_metadata_filename: str, in_vecs_filename: str, out_vecs_filename: str, out_metadata_filename: str, delim=",") -> None:
    metadata_rows = []
    vec_rows = []
    with open(in_metadata_filename, "r") as metadata_f, open(in_vecs_filename, "r") as vecs_f:
        reader_m = csv.reader(metadata_f, delimiter=delim)
        reader_v = csv.reader(vecs_f, delimiter=delim)

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


def clean_all_in_dir(dirname: str, out_dir: str, stopwords: list, lemmatizer: WordNetLemmatizer, get_abstract: bool = False) -> None:
    for filename in os.listdir(dirname):
        print(filename)
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            if not os.path.exists(new_path):
                preproc_pubs(full_path, stopwords, lemmatizer, new_path, get_abstract=get_abstract)


def make_pubs_vectors_in_dir(dirname: str, out_dir: str, vecs: Dict, has_abstract: bool = False, abstract_only: bool = False) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            make_pubs_vectors(full_path, vecs, new_path, has_abstract=has_abstract, abstract_only=abstract_only)

def split_in_dir(dirname: str, out_dir_1: str, out_dir_2: str) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            outpath_1 = os.path.join(out_dir_1, filename)
            outpath_2 = os.path.join(out_dir_2, filename)
            split_file(full_path, outpath_1, outpath_2)

def make_tsv_files_in_dir(in_dir_metadata: str, in_dir_vecs: str, out_dir_metadata: str, out_dir_vecs: str) -> None:
    for filename in os.listdir(in_dir_metadata):
        if filename.endswith(".csv"):
            metadata_in = os.path.join(in_dir_metadata, filename)
            vec_in = os.path.join(in_dir_vecs, filename)
            metadata_out = os.path.join(out_dir_metadata, filename)
            vec_out = os.path.join(out_dir_vecs, filename)
            make_tsv_files(metadata_in, vec_in, vec_out, metadata_out)

if __name__ == "__main__":
    make_tsv_files("./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-labels-final.csv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-300-final.csv",
    "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.tsv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-labels-final.tsv")
    assert False
    split_file("./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-labels-final.csv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-300-final.csv")
    assert False
    """make_tsv_files("./data/turing_winners/vecs-abstracts-ordered/Geoff-Hinton-abstract-labels.tsv",
    "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs.tsv", 
    "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstracts-only-tb.tsv",
    "./data/turing_winners/abstracts/geoff/Geoff-Hinton-labels-tb.tsv", 
    delim="\t")
    assert False"""
    lemmatizer = WordNetLemmatizer()
    stopwords = stopwords.words("english")
    vecs = load_vectors("./data/external/wiki-news-300d-1M.vec")

    #clean_all_in_dir("./data/turing_winners/abstracts", "./data/turing_winners/abstracts-cleaned", stopwords, lemmatizer, get_abstract=True)
    make_pubs_vectors("./data/turing_winners/abstracts/geoff/Geoff-Hinton-cleaned.csv", vecs, "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", has_abstract=True, abstract_only=True)
    assert False
    make_pubs_vectors_in_dir("./data/turing_winners/abstracts-cleaned", "./data/turing_winners/vecs-abstracts-n", vecs, has_abstract=True, abstract_only=True)
    assert False
    """make_pubs_vectors(
        "./data/turing_winners/cleaned/Adi-Shamir.csv",
        vecs,
        "./data/turing_winners/vecs/Adi-Shamir.csv")"""

    """make_tsv_files(
        "./data/hinton_titles_selected.csv",
        "./data/hinton_vecs_selected.csv",
        "./data/hinton_vecs_selected.tsv",
        "./data/hinton_titles_selected.tsv")"""
