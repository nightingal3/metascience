import csv
import os
import re
import time
from typing import List
import pdb

from bs4 import BeautifulSoup
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait

from scrape_from_table import get_table_cells

timeout = 8

def scrape_birth_death_years(url: str, out_filename: str, name_col: List = [2]) -> None:
    driver = webdriver.Chrome(executable_path="./chromedriver.exe")
    driver.get(url)
    driver.set_page_load_timeout(timeout)
    birth_xpath = "//table[contains(@class,'infobox') and contains(@class, 'vcard')]/tbody/tr/th[contains(text(), 'Born')]/following-sibling::td"
    death_xpath = "//table[contains(@class,'infobox') and contains(@class, 'vcard')]/tbody/tr/th[contains(text(), 'Died')]/following-sibling::td"
    out_rows = [["name","birth_date","death_date"]]

    table_rows = get_table_cells(url, name_col)
    for row in table_rows:
        try:
            driver.implicitly_wait(timeout)
            name = row[0].a.text
            print(name)
            link = WebDriverWait(driver, timeout).until(lambda d: d.find_element_by_link_text(name))
            link.click()
            driver.implicitly_wait(timeout)
            _ = WebDriverWait(driver, timeout).until(lambda d: d.find_element_by_xpath(birth_xpath))
            birth_text = driver.find_elements_by_xpath(birth_xpath)
            if len(birth_text) > 0:
                birth_date = re.search("\d{4}", birth_text[0].text).group(0)
            else: 
                birth_date = "N/A"
            death_text = driver.find_elements_by_xpath(death_xpath)
            if len(death_text) > 0:
                death_date = re.search("\d{4}", death_text[0].text).group(0)
            else:
                death_date = "N/A"
            print(birth_date)
            print(death_date)
            out_rows.append([name, birth_date, death_date])
            driver.back()
        except Exception as e:
            print(e)
            out_rows.append([name, birth_date, death_date])
            driver.back()

    with open(out_filename, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(out_rows)

def filter_all_names(paper_dir: str, names_filename: str, out_dir: str) -> None:
    with open(names_filename, "r") as names_file: 
        reader = csv.reader(names_file)
        lines = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            lines.append(row)

    for line in lines:
        name, birth_year, death_year = line
        print(name)
        birth_year = int(birth_year) if birth_year != "N/A" else 0
        death_year = int(death_year) if death_year != "N/A" else float("inf")
        scientist_filename = f"{paper_dir}/{name}.csv"
        out_filename = f"{out_dir}/{name}.csv"

        if os.path.isfile(scientist_filename):
            try:
                df = pd.read_csv(scientist_filename, header=None)
            except pd.errors.EmptyDataError:
                continue
            df.columns = [i for i in range(len(df.columns))]
            filtered_df = filter_by_year(df, birth_year, death_year, 0)
            if filtered_df.empty:
                continue
            filtered_df.to_csv(out_filename, header=False, index=False)

def filter_by_year(df: pd.DataFrame, filter_start_year: int, filter_end_year: int, year_col: int) -> None:
    filtered = df[(df[year_col] <= filter_end_year) & (df[year_col] >= filter_start_year + 15)] # likely wouldn't publish papers before 15
    #filtered = df[by_year]
    return filtered


if __name__ == "__main__":
    #url = "https://en.wikipedia.org/wiki/List_of_Nobel_Memorial_Prize_laureates_in_Economics"
    filter_all_names("./data/nobel_winners/physics/abstracts-fixed", "./data/nobel_winners/physics/birth_death_dates.csv", "./data/nobel_winners/physics/abstracts_filtered_year")
    #scrape_birth_death_years(url, "./data/nobel_winners/economics/birth_death_dates.csv")
