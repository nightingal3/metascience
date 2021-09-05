import csv
from math import floor
import os
import pickle
import pdb

from pandas import read_csv, DataFrame

from make_pie_chart import get_winners, make_pie_chart

decade_groups = {
    "cs-rand": [1920, 1980, 2000],
    "chemistry-rand": [1940, 1980, 2000],
    "economics-rand": [1920, 1980, 2000],
    "medicine-rand": [1940, 1980, 2000],
    "physics-rand": [1920, 1980, 2000]
}

def split_scientists_earliest_paper(scientist_dir: str, field: str) -> dict:
    """Split scientists into decade groups based on their earliest paper.

    Args:
        scientist_dir (str): Directory of scientist abstracts

    Returns:
        dict: {decade: [scientist names]}
    """
    decades = {}
    for filename in os.listdir(scientist_dir):
        if not filename.endswith(".csv"):
            continue
        df = read_csv(f"{scientist_dir}/{filename}", names=["year", "title", "abstract"], header=None)
        if len(df) == 0:
            continue

        min_year = min(df.year)
        for group in sorted(decade_groups[field], reverse=True):
            if min_year > group:
                if group in decades:
                    decades[group].append(filename[:-4])
                else:
                    decades[group] = [filename[:-4]]   
                break  
    
    return decades

def pie_chart_decades(decades_dict: dict, models_dict: dict) -> None:
    for decade in decades_dict:
        selected_scientists = {model_name: {} for model_name in models_dict}
        for model_name in models_dict:
            for scientist_name in models_dict[model_name]:
                if scientist_name in decades_dict[decade]:
                    if model_name != "Null" and models_dict[model_name][scientist_name][1] < 5:
                        continue
                    selected_scientists[model_name][scientist_name] = models_dict[model_name][scientist_name]
        print(decade)
        print(len(selected_scientists["kNN"]))
        make_pie_chart(get_winners(selected_scientists, p_vals=False, include_null=True, len_included=True, dict_format=False), include_null=True, filename=f"phys-random-decade-{decade}-fixed")

if __name__ == "__main__":
    decades_names = split_scientists_earliest_paper("./data/nobel_winners/physics-random/abstracts-cleaned", field="physics-rand")
    models = pickle.load(open("results/full-fixed/physics-random.p", "rb"))
    pie_chart_decades(decades_names, models)


        