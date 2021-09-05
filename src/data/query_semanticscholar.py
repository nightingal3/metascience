import csv
import os
import sys
import time
from typing import List

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
import pdb

url = "https://www.semanticscholar.org/search?publicationType%5B0%5D=JournalArticle&q="
sort_by = "&sort=pub-date"
timeout = 8
scroll_pause_time = 0.25

def scrape_from_semanticscholar(name: str, filename: str, get_abstracts: bool = False, get_authorship: bool = False) -> None:
    req = requests.get(url + name + sort_by)
    data = req.text
    
    soup = BeautifulSoup(data, features="html.parser")
    print(soup)
    entries = [soup.findAll("article", {"class": "search-result"})]
    print("entries: ", entries)

    titles = []
    abstracts = []

    for i, e in enumerate(entries):
        title = e.find("div", {"class": "search-result-title"}).text
        year = e.find("span", {"data-selenium-selector": "paper-year"}).text
        print(title, year)


def get_scientist_papers(in_filename: str, out_dir: str, authorship_dir: str, authorship_only: bool = False) -> None:
    with open(in_filename, "r") as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if len(row) == 2:
                name, scientist_id = row
                #scientist_id, name = row
            else:
                scientist_id = row[0]

            if scientist_id == "N/A":
                continue

            print(scientist_id)
            if len(row) == 2:
                if os.path.isfile(f"{authorship_dir}/{name}.csv"):
                    print("Already done")
                    continue
                get_all_papers(scientist_id, f"{out_dir}/{name}.csv", f"{authorship_dir}/{name}.csv", authorship_only)
            else:
                if os.path.isfile(f"{authorship_dir}/{scientist_id}.csv"):
                    print("Already done")
                    continue
                get_all_papers(scientist_id, f"{out_dir}/{scientist_id}.csv", f"{authorship_dir}/{scientist_id}.csv", authorship_only)     

def get_selected_scientist_papers(sbert_dir: str, out_dir: str) -> None:
    done_files = list(os.listdir(out_dir))
    for filename in os.listdir(sbert_dir):
        if filename.endswith(".csv"):
            scientist_id = filename[:-4]
            print(scientist_id)

            if filename in done_files:
                print("Already done")
                continue

            get_all_papers(scientist_id, f"{out_dir}/abstracts/{scientist_id}.csv", f"{out_dir}/{scientist_id}.csv", authorship_only=True)


def delete_nonselected_papers(sbert_dir: str, out_dir: str) -> None:
    selected_files = list(os.listdir(sbert_dir))
    for filename in os.listdir(out_dir):
        if filename.endswith(".csv"):
            full_path = os.path.join(out_dir, filename)
            if filename not in selected_files:
                os.remove(full_path)


def get_all_papers(scientist_id: str, filename: str, authorship_filename: str, authorship_only: bool = False) -> None:
    author_url = "https://api.semanticscholar.org/v1/author/" + scientist_id
    req = requests.get(author_url)
    data = req.json()
    papers = None
    try:
        papers = data["papers"]
    except KeyError:
        if "error" in data:
            print("Author not found or other error")
            return
        time.sleep(30)  # rate limited 100 reqs/5 mins
        get_all_papers(scientist_id, filename, authorship_filename)

    years = []
    titles = []
    abstracts = []
    num_authors = []
    is_first_author = []
    
    
    if not criterion(papers):
        return
    for paper in papers:
        if paper["year"] is None:
            continue
        try:
            years.append(paper["year"])
        except:
            continue
        titles.append(paper["title"])
        print(paper["title"])
       
        abstract, num_author, first_author = get_paper_info(paper["paperId"], scientist_id)
        abstracts.append(abstract)
        num_authors.append(num_author)
        is_first_author.append(first_author)
    
    if not authorship_only:
        with open(filename, "w") as out_file:
            writer = csv.writer(out_file)
            for i, year in enumerate(years):
                writer.writerow([year, titles[i], abstracts[i]])
        
    with open(authorship_filename, "w") as author_file:
        writer = csv.writer(author_file)
        for i, title in enumerate(titles):
            writer.writerow([title, num_authors[i], is_first_author[i]])


def get_paper_info(paper_id: str, author_id: str) -> List:
    paper_url = "https://api.semanticscholar.org/v1/paper/" + paper_id
    num_authors, first_author = 0, 0
    try:
        req = requests.get(paper_url).json()
        if "message" in req:
            if req["message"] == "Internal server error" or req["message"] == "Endpoint request timed out":
                return ["", 0, 0]
            time.sleep(30)
            abstract = get_paper_info(paper_id, author_id)
        if "abstract" not in req or req["abstract"] is None:
            abstract = ""
        else:
            abstract = req["abstract"]
        if "authors" in req:
            num_authors, first_author = get_authorship(req["authors"], author_id)
    except:
        time.sleep(30)  # rate limited 100 reqs/5 mins
        abstract = get_paper_info(paper_id, author_id)
    
    return [abstract, num_authors, first_author]

def get_authorship(author_list: List, author_id: str) -> tuple:
    num_authors = len(author_list)
    if num_authors == 0:
        return 0, 0
    is_first_author = author_id == author_list[0]["authorId"] or author_id == author_list[-1]["authorId"]
    return num_authors, int(is_first_author)
        

def criterion(paper_list: List) -> bool:
    return paper_list is not None and len(paper_list) >= 5


if __name__ == "__main__":
    field = "physics"
    #delete_nonselected_papers(f"data/nobel_winners/chemistry/random-sample/sbert-abstracts-ordered", f"data/nobel_winners/chemistry/random-sample/authorship")

    #delete_nonselected_papers(f"data/nobel_winners/{field}/random-sample/sbert-abstracts-ordered", f"data/nobel_winners/{field}/random-sample/authorship")
    #get_scientist_papers("./data/nobel_winners/random_biologists.txt", "./data/nobel_winners/physics/random-sample/abstracts", "./data/nobel_winners/physics/random-sample/authorship")
    #get_selected_scientist_papers(f"data/nobel_winners/{field}/random-sample/sbert-abstracts-ordered", f"data/nobel_winners/{field}/random-sample/authorship")
    #get_selected_scientist_papers(f"data/nobel_winners/chemistry/random-sample/sbert-abstracts-ordered", f"data/nobel_winners/chemistry/random-sample/authorship")

    get_scientist_papers("./data/nobel_winners/medicine/name_to_id.csv", "./data/nobel_winners/medicine/abstracts-fixed", "./data/nobel_winners/medicine/authorship-fixed")
    #get_all_papers("50702974", "data/nobel_winners/physics/authorship/abstracts/Albert Einstein 2.csv", "data/nobel_winners/chemistry/authorship/Albert Einstein 2.csv")


    


