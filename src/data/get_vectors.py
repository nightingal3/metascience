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

from langdetect import detect
import pandas as pd
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


def preproc_pubs(filename: str, stop_words: list, lemmatizer: WordNetLemmatizer, out_filename: str, get_abstract: bool = False, split_abstract: bool = False, authorship: bool = False) -> None:
    cleaned_rows = []
    titles_seen = set()
    with open(filename, "r", encoding="utf-8", errors="ignore") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if get_abstract:
                year, title_ch, abstract = row
                if year == "" or abstract == "":
                    continue
                try:
                    if detect(title_ch) != "en" or detect(abstract) != "en":
                        continue
                except:
                    continue
            else:
                if authorship:
                    title_ch, rest = row[0], row[1:]
                    pdb.set_trace()
                else:
                    year, title_ch, rest = row[0], row[1], row[2:]
                try:
                    if detect(title_ch) != "en":
                        continue
                except:
                    continue

            title_words = (
                "".join(
                    ch for ch in title_ch if ch not in punctuation)).lower().split(" ")
            # papers with the same name/content may appear multiple times
            if "".join(title_words) in titles_seen: 
                continue

            clean_title = " ".join([lemmatizer.lemmatize(word)
                                    for word in title_words if word not in stop_words])

            clean_title = title_ch

            if get_abstract:
                cleaned_abstract = clean_abstract(abstract)
                if split_abstract:
                    abstract_sentences = abstract.split(".")
                    lemmatized_abstract_split = []
                    for sent in abstract_sentences:
                        clean_sent = "".join(ch for ch in sent if ch not in punctuation).lower().split(" ")
                        lemmatized_abstract_split.append(" ".join([lemmatizer.lemmatize(word) for word in clean_sent if word not in stop_words]))
                    lemmatized_abstract = ".".join(lemmatized_abstract_split)
                else:
                    abstract_words = "".join(
                        ch for ch in cleaned_abstract if ch not in punctuation).lower().split(" ")
                    lemmatized_abstract = " ".join([lemmatizer.lemmatize(word)
                                        for word in abstract_words if word not in stop_words])

            if get_abstract:
                cleaned_rows.append([year, clean_title, lemmatized_abstract])
            else:
                if authorship:
                    cleaned_rows.append([clean_title, *rest])
                else:
                    cleaned_rows.append([year, clean_title, *rest])

            titles_seen.add("".join(title_words))

    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        for row in cleaned_rows:
            writer.writerow(row)


def get_author_labels(titles_filename: str, authorship_filename: str, out_filename: str) -> None:
    titles = pd.read_csv(titles_filename, header=None)
    authors = pd.read_csv(authorship_filename, header=None)
    print(authorship_filename)

    merged = pd.merge(titles, authors, left_on=1, right_on=0, how="left")
    #merged = merged[[0, 1, "2_y", "3_y"]]
    #pdb.set_trace()
    merged = merged[["0_x", 1, "1_y", "2_y", "2_x"]]

    merged = merged.dropna(subset=["0_x", "1_y", "2_y"])
    merged = merged[merged["1_y"] != 0] # 0 authors = couldn't find info
    merged["1_y"] = merged["1_y"].dropna()
    merged["2_y"] = merged["2_y"].dropna()

    merged["0_x"] = merged["0_x"].astype(int)
    merged["1_y"] = merged["1_y"].astype(int)
    merged["2_y"] = merged["2_y"].astype(int)

    merged.to_csv(out_filename, header=False, index_label=False, index=False)

def join_cells(filename: str, out_filename: str, start_col: int = 2) -> None:
    joined_rows = []
    with open(filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            joined_abstract = ", ".join(row[start_col:])
            joined_rows.append([*row[:start_col], joined_abstract])

    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        for row in joined_rows:
            writer.writerow(row)

def join_cells_in_dir(in_dir: str, out_dir: str, start_col: int = 2) -> None:
    for filename in os.listdir(in_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(in_dir, filename)
            new_path = os.path.join(out_dir, filename)
            join_cells(full_path, new_path, start_col)

def join_csv_cols(filename1: str, filename2: str, vecs_col1: int, vecs_col2: int, out_filename: str) -> None:
    with open(filename1, "r") as file1, open(filename2, "r") as file2:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)
        rows1 = []
        rows2 = []

        for row in reader1:
            rows1.append([row[0], row[1], ", ".join(row[vecs_col1:])])
        for i, row in enumerate(reader2):
            #rows2.append([row[0], row[1], ", ".join(row[vecs_col2:])])

            rows2.append([row[0], row[1], row[2]])
        
        df1 = pd.DataFrame(rows1)
        df2 = pd.DataFrame(rows2)
        new_df = pd.merge(df1, df2,  how='inner', on=[0, 1])

        new_df.to_csv(out_filename, header=False)


def split_file(filename: str, out_filename_0: str, out_filename_1: str, multicols: bool = False) -> None:
    f_0 = []
    f_1 = []
    with open(filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            f_0.append([row[0], row[1]])
            if multicols:
                f_1.append([row[0], ", ".join(row[2:])])
            else:
                f_1.append([row[0], row[2]])
    
    with open(out_filename_0, "w") as out_f_0:
        writer = csv.writer(out_f_0)
        for row in f_0:
            writer.writerow(row)
    
    with open(out_filename_1, "w") as out_f_1:
        writer = csv.writer(out_f_1)
        for row in f_1:
            writer.writerow(row)

def rename_files(id_to_name_mapping_filepath: str, dir_to_be_renamed: str) -> None:
    mapping = {}
    with open(id_to_name_mapping_filepath, "r") as name_to_id_f:
        reader = csv.reader(name_to_id_f)
        for row in reader:
            mapping[row[1]] = row[0]
    
    for filename in os.listdir(dir_to_be_renamed):
        src = os.path.join(dir_to_be_renamed, filename)
        dst = os.path.join(dir_to_be_renamed, mapping[filename[:-4]] + ".csv")
        
        os.rename(src, dst)

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
                    vec_rows.append([year, " ".join(title), avg_vec])
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


def clean_all_in_dir(dirname: str, out_dir: str, stopwords: list, lemmatizer: WordNetLemmatizer, get_abstract: bool = False, split_abstract: bool = False, authorship: bool = False) -> None:
    for filename in os.listdir(dirname):
        print(filename)
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            if not os.path.exists(new_path):
                preproc_pubs(full_path, stopwords, lemmatizer, new_path, get_abstract=get_abstract, split_abstract=split_abstract, authorship=authorship)


def make_pubs_vectors_in_dir(dirname: str, out_dir: str, vecs: Dict, has_abstract: bool = False, abstract_only: bool = False) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            new_path = os.path.join(out_dir, filename)
            make_pubs_vectors(full_path, vecs, new_path, has_abstract=has_abstract, abstract_only=abstract_only)

def split_in_dir(dirname: str, out_dir_1: str, out_dir_2: str, multicols: bool = False) -> None:
    for filename in os.listdir(dirname):
        if filename.endswith(".csv"):
            full_path = os.path.join(dirname, filename)
            outpath_1 = os.path.join(out_dir_1, filename)
            outpath_2 = os.path.join(out_dir_2, filename)
            split_file(full_path, outpath_1, outpath_2, multicols=multicols)

def make_tsv_files_in_dir(in_dir_metadata: str, in_dir_vecs: str, out_dir_metadata: str, out_dir_vecs: str) -> None:
    for filename in os.listdir(in_dir_metadata):
        if filename.endswith(".csv"):
            metadata_in = os.path.join(in_dir_metadata, filename)
            vec_in = os.path.join(in_dir_vecs, filename)
            metadata_out = os.path.join(out_dir_metadata, filename)
            vec_out = os.path.join(out_dir_vecs, filename)
            make_tsv_files(metadata_in, vec_in, vec_out, metadata_out)

def get_author_labels_in_dir(titles_dir: str, authors_dir: str, out_dir: str) -> None:
    for filename in os.listdir(titles_dir):
        if filename.endswith(".csv"):
            titles = os.path.join(titles_dir, filename)
            authors = os.path.join(authors_dir, filename)
            out = os.path.join(out_dir, filename)
            if os.path.isfile(titles) and os.path.isfile(authors):
                get_author_labels(titles, authors, out)

def join_csv_cols_in_dir(dir1: str, dir2: str, out_dir: str) -> None:
    for filename in os.listdir(dir1):
        if ".csv" in filename:
            file1 = os.path.join(dir1, filename)
            file2 = os.path.join(dir2, filename)
            out = os.path.join(out_dir, filename)
            join_csv_cols(file1, file2, 2, 2, out)

if __name__ == "__main__":
    #join_csv_cols("data/others/sbert-abstracts/Yang-Xu.csv", "data/others/vecs-titles-w-labels/Yang-Xu.csv", 2, 1, "data/others/sbert-abstracts-vs-fasttext-titles/Yang-Xu.csv")
    #assert False
    """join_csv_cols_in_dir("data/turing_winners/sbert-titles/", "data/turing_winners/sbert-abstracts", "data/turing_winners/sbert-titles-and-abstracts/")
    assert False
    split_in_dir("data/turing_winners/sbert-abstracts", "data/turing_winners/sbert-labels", "data/turing_winners/sbert-vecs-only", multicols=True)
    make_tsv_files_in_dir("data/turing_winners/sbert-labels", "data/turing_winners/sbert-vecs-only", "data/turing_winners/sbert-labels-tsv", "data/turing_winners/sbert-vecs-tsv")
    assert False"""
    
    #join_cells_in_dir("./data/nobel_winners/chemistry/sbert-abstracts/", "./data/nobel_winners/chemistry/sbert-abstracts-2")
    #rename_files("data/nobel_winners/physics/name_to_id.csv", "data/nobel_winners/physics/authorship")
    get_author_labels_in_dir("data/nobel_winners/chemistry/sbert-abstracts-2", "data/nobel_winners/chemistry/authorship", "data/nobel_winners/chemistry/sbert-labels-and-authors")
    assert False
    #get_author_labels("data/turing_winners/vecs-abstracts-w-labels/abstract-labels/Adi-Shamir.csv", "data/turing_winners/authorship/Adi-Shamir.csv", "data/turing_winners/vecs-abstracts-w-labels/labels-and-authors/Adi-Shamir.csv")
    #split_file("./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-labels-final.csv", "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-300-final.csv")
    lemmatizer = WordNetLemmatizer()
    stopwords = stopwords.words("english")
    #vecs = load_vectors("./data/external/wiki-news-300d-1M.vec")
    clean_all_in_dir("./data/nobel_winners/chemistry/authorship", "./data/nobel_winners/chemistry/authorship-cleaned", stopwords, lemmatizer, get_abstract=False, authorship=True)
    #make_pubs_vectors("./data/others/abstracts/geoff/Geoff-Hinton-cleaned.csv", vecs, "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.csv", has_abstract=True, abstract_only=True)
    #smake_pubs_vectors_in_dir("./data/others/abstracts-cleaned", "./data/others/vecs-titles-w-labels", vecs, has_abstract=True, abstract_only=False)