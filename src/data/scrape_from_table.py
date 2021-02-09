import argparse
import csv
from typing import List

from bs4 import BeautifulSoup
import requests

def scrape_from_table(url: str, filename: str, cols=[]) -> None:
    rows_out = []
    table_rows = get_table_cells(url, cols)
    for row in table_rows: 
        print(row)
        try:
            rows_out.append([cell.a.text for cell in row])
        except: 
            continue
    
    with open(filename, "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(rows_out)
    

def get_table_cells(url: str, cols: List = []) -> List:
    req = requests.get(url)
    data = req.text
    soup = BeautifulSoup(data, features="html.parser")
    rows_out = []

    table = soup.find("table")
    rows = table.find_all("tr")
    num_cols = len(rows[0].find_all({"td": True, "th": True}))

    for row in rows[1:]:
        d = row.find_all({"td": True, "th": True})
        rows.append(d)
        try: 
            if len(d) == 2:
                continue 
            inds = [i for i in cols] if len(d) == num_cols else [i - 1 for i in cols]
            rows_out.append([d[i] for i in inds])
        except:
            continue
    return rows_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", help="the url of a site containing the table to scrape", required=True)
    parser.add_argument("-o", "--out", help="a file to output the scraped table to ", default="./table.csv")
    parser.add_argument("-c", "--cols", help="columns to scrape from", nargs="*", default="2")
    args = parser.parse_args()

    if args.cols is not None:
        cols = [int(c) for c in args.cols]
        print("cols: ", cols)
        scrape_from_table(args.url, args.out, cols)
    else:
        scrape_from_table(args.url, args.out)



