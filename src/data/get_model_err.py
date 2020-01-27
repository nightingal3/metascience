import csv
import inspect
import pdb

import src.models.predict as predict
from src.models.ranking_functions import rank_on_1NN

def get_model_rank_err(model_name: str, out_filename: str = "data/model_rank_err.csv") -> None:
    print(rank_on_1NN)
    try:
        with open(out_filename, "a") as err_file:
            writer = csv.writer(err_file)
            try:
                #ranking_func = getattr(rank_funcs, model_name)
                ranking_func = rank_on_1NN
                print(ranking_func)
            except:
                print("Not a valid ranking function, see models/ranking_functions.py")
    except IOError:
        print("IOError")

if __name__ == "__main__":
    get_model_rank_err("rank_on_prototype")
