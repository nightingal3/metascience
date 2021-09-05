import numpy as np
import pickle
import pdb

from src.visualizations.make_pie_chart import make_pie_chart

def calc_aic(N: int, log_L: float, num_params: int) -> float:
    return 2 * num_params - 2 * log_L

def calc_bic(N: int, log_L: float, num_params: int) -> float:
    return -2 * log_L + np.log(N) * num_params

def calc_aic_bic_individuals(model_data: dict) -> tuple:
    if "2NN" in model_data:
        del model_data["2NN"]
    if "3NN" in model_data:
        del model_data["3NN"]
    if "4NN" in model_data:
        del model_data["4NN"]
    if "5NN" in model_data:
        del model_data["5NN"]
    if "Exemplar (s=0.001)" in model_data:
        del model_data["Exemplar (s=0.001)"]
    if "Exemplar (s=0.1)" in model_data:
        del model_data["Exemplar (s=0.1)"]
    if "Exemplar (s=1)" in model_data:
        del model_data["Exemplar (s=1)"]

    individual_aic = {}
    individual_bic = {}
    for name in model_data["1NN"]:
        null_LL = model_data["Null"][name]
        num_papers = model_data["1NN"][name][1]

        if num_papers < 5:
            continue

        model_LL = {model_name: model_data[model_name][name][0] + null_LL for model_name in model_data if model_name != "Null"}
        aic = {model_name: calc_aic(num_papers, LL, 0) for model_name, LL in model_LL.items() if "Exemplar" not in model_name}
        aic["Exemplar"] = calc_aic(num_papers, model_LL["Exemplar"], 1)
        aic["Null"] = calc_aic(num_papers, null_LL, 0)
        #aic["Exemplar (s=0.001)"] = calc_aic(num_papers, model_LL["Exemplar (s=0.001)"], 1)
        #aic["Exemplar (s=0.1)"] = calc_aic(num_papers, model_LL["Exemplar (s=0.1)"], 1)
        #aic["Exemplar (s=1)"] = calc_aic(num_papers, model_LL["Exemplar (s=1)"], 1)


        bic = {model_name: calc_bic(num_papers, LL, 0) for model_name, LL in model_LL.items() if "Exemplar" not in model_name}
        bic["Exemplar"] = calc_bic(num_papers, model_LL["Exemplar"], 1)
        bic["Null"] = calc_bic(num_papers, null_LL, 0)

        #bic["Exemplar (s=0.001)"] = calc_bic(num_papers, model_LL["Exemplar (s=0.001)"], 1)
        #bic["Exemplar (s=0.1)"] = calc_bic(num_papers, model_LL["Exemplar (s=0.1)"], 1)
        #bic["Exemplar (s=1)"] = calc_bic(num_papers, model_LL["Exemplar (s=1)"], 1)

        individual_aic[name] = aic
        individual_bic[name] = bic

    return individual_aic, individual_bic

def get_best_model_individual(individual_data: dict) -> dict:
    best_per_individual = {name: [float("inf"), []] for name in individual_data}
    for individual in individual_data: 
        for model in individual_data[individual]:
            if individual_data[individual][model] < best_per_individual[individual][0]:
                best_per_individual[individual][0] = individual_data[individual][model]
                best_per_individual[individual][1] = [model]
            elif individual_data[individual][model] == best_per_individual[individual][0]:
                best_per_individual[individual][1].append(model)

    return best_per_individual


def calc_aic_bic_overall(model_data: dict) -> dict:
    if "2NN" in model_data:
        del model_data["2NN"]
    if "3NN" in model_data:
        del model_data["3NN"]
    if "4NN" in model_data:
        del model_data["4NN"]
    if "5NN" in model_data:
        del model_data["5NN"]

    aic = {}
    bic = {}
    num_papers_overall = 0
    log_L_overall = {model_name: 0 for model_name in model_data}
    for name in model_data["1NN"]:
        null_LL = model_data["Null"][name]

        num_papers = model_data["1NN"][name][1]
        num_papers_overall += num_papers

        for model_name in model_data:
            if model_name == "Null":
                continue
            model_LL = model_data[model_name][name][0] + null_LL
            log_L_overall[model_name] += model_LL

    for model_name in model_data:
        num_params = 1 if "Exemplar" in model_name else 0
        aic[model_name] = calc_aic(num_papers_overall, log_L_overall[model_name], num_params)
        bic[model_name] = calc_bic(num_papers_overall, log_L_overall[model_name], num_params)

    return aic, bic

if __name__ == "__main__":
    field = "medicine"
    num_authors = 2
    #model_data = pickle.load(open(f"results/full-2/{field}.p", "rb"))
    model_data = pickle.load(open(f"results/summary/k-author/authorship-{field}-{num_authors}.p", "rb"))
    #model_data = pickle.load(open(f"results/summary/k-author/authorship-{field}-{num_authors}.p", "rb"))
    aic, bic = calc_aic_bic_individuals(model_data)
    aic_individual = get_best_model_individual(aic)
    bic_individual = get_best_model_individual(bic)

    make_pie_chart(aic_individual, include_null=True, len_included=False, filename=f"aic-{field}-{num_authors}")
    make_pie_chart(bic_individual, include_null=True, len_included=False, filename=f"bic-{field}-{num_authors}")
    #print(aic)
    #print(bic)
    #print(get_best_model_individual(bic))