import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from collections import OrderedDict
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import pdb
import operator

order = OrderedDict({
    "1NN": 0,
    #"2NN": 1,
    #"3NN": 2,
    #"4NN": 3,
    #"5NN": 4, 
    "Exemplar": 5,
    "Exemplar (s=0.001)": 6,
    "Exemplar (s=0.1)": 7,
    "Exemplar (s=1)": 8,
    "Exemplar":9,
    "Progenitor": 10,
    "Prototype": 11,
    #"Exemplar (s=1)": 8,
    "Local": 12,
    "Null": 13
})

colours = dict(zip(order.keys(), plt.cm.Set3.colors[:len(order)]))

def get_winners(model_dict: dict, p_vals=True, len_included=False, include_null=False) -> dict:
    del model_dict["Exemplar (s=1)"] # no longer including base exemplar
    del model_dict["Exemplar (s=0.1)"]
    del model_dict["Exemplar (s=0.001)"]
    del model_dict["2NN"]
    del model_dict["3NN"]
    del model_dict["4NN"]
    del model_dict["5NN"]
    #if not include_null:
        #del model_dict["Null"]
    scientists_to_models = {k: (-float("inf"), None) for k in model_dict[list(model_dict.keys())[0]]}

    if len_included:
        for model in model_dict:
            for scientist in model_dict[model]:
                if p_vals and model_dict[model][scientist][1] * len(scientists_to_models) > 0.05:
                    continue
                if p_vals:
                    val = model_dict[model][scientist][0]
                else:
                    if model_dict[model][scientist][1] < 5:
                        continue
                    val = model_dict[model][scientist][0]
                if val > scientists_to_models[scientist][0]:
                    scientists_to_models[scientist] = (val, [model])
                elif val == scientists_to_models[scientist][0]:
                    scientists_to_models[scientist][1].append(model)
        if include_null:
            scientists_to_models = {k: v if v[0] > 0 else (0, ["Null"]) for k, v in scientists_to_models.items()}
    else:
        for model in model_dict:
            for scientist in model_dict[model]:
                val = model_dict[model][scientist]
                if val > scientists_to_models[scientist][0]:
                    scientists_to_models[scientist] = (val, [model])
                elif val == scientists_to_models[scientist][0]:
                    scientists_to_models[scientist][1].append(model)
        if include_null:
            scientists_to_models = {k: v if v[0] > 0 else (0, ["Null"]) for k, v in scientists_to_models.items()}
    return scientists_to_models

def get_pairwise_differences(model_dict: dict, p_vals=True, include_null=True) -> dict:
    if not include_null:
        del model_dict["Null"]

    scientists_to_models = {k: [] for k in model_dict['1NN']}
    for model in model_dict:
        for scientist in model_dict[model]:
            scientists_to_models[scientist].append((model_dict[model][scientist], model))
    
    gaps = []
    all_1NN = []
    all_progenitor = []

    for scientist in scientists_to_models:
        model_performance = sorted([i for i in scientists_to_models[scientist]], key=lambda x: x[0], reverse=True)
        gaps.append((model_performance[0][0][0] - model_performance[1][0][0], model_performance[0][1], scientist))
        #print(scientists_to_models[scientist])
        all_1NN.append((scientists_to_models[scientist][0][0][0], scientists_to_models[scientist][0][0][1]))
        all_progenitor.append((scientists_to_models[scientist][8][0][0], scientists_to_models[scientist][8][0][1]))
    
    advantage_1NN = sorted([i for i in gaps if i[1] == "1NN"], key=lambda x:x[0], reverse=True)
    advantage_progenitor = sorted([i for i in gaps if i[1] == "Prototype"], key=lambda x:x[0], reverse=True)

    rank_1NN, scientists = zip(*(sorted(all_1NN, key=lambda x:x[0])))
    print(advantage_1NN[:5])
    print(advantage_progenitor[:5])
    rank_1NN, num_papers = zip(*all_1NN)
    print(rank_1NN)
    print(num_papers)
    rank_prog, num_papers_p = zip(*all_progenitor)
    print(pearsonr(num_papers, rank_1NN))
    print(pearsonr(num_papers_p, rank_prog))
    plt.scatter(num_papers, rank_prog)
    plt.savefig("num_papers_vs_prototype_dominance")
    assert False

    return advantage_1NN, advantage_progenitor

def make_pie_chart(scientists_to_models, include_null=True, len_included=False, filename="trial-pie") -> dict:
    percentages = {}
    plt.gcf().clear()

    for scientist in scientists_to_models:
        if scientists_to_models[scientist][1] is None:
            continue
        if not include_null and scientists_to_models[scientist][1][0] == "Null":
            continue
        if len(scientists_to_models[scientist][1]) == 1:
            if scientists_to_models[scientist][1][0] not in percentages:
                percentages[scientists_to_models[scientist][1][0]] = 1
            else:
                percentages[scientists_to_models[scientist][1][0]] += 1
        else:
            for mod in scientists_to_models[scientist][1]:
                if mod not in percentages:
                    percentages[mod] = 1/len(scientists_to_models[scientist][1])
                else:
                    percentages[mod] += 1/len(scientists_to_models[scientist][1])

    
    num_not_null = len(scientists_to_models)
    if not include_null:
        num_not_null = len([v for k, v in scientists_to_models.items() if  v[1] is not None and v[1][0] != "Null"])

    percentages = {k: v / num_not_null for k, v in percentages.items()}
    #if "Exemplar" in percentages:
        #percentages["Exemplar (s=0.01)"] = percentages["Exemplar"]
        #del percentages["Exemplar"]

    print(sorted(list(percentages.items()), key=operator.itemgetter(1), reverse=True))
    percent_sorted = sorted(list(percentages.items()), key=lambda x: order[x[0]])
    plt.pie([i[1] for i in percent_sorted], labels=[i[0] for i in percent_sorted], autopct='%1.1f%%', colors=[colours[key[0]] for key in percent_sorted], pctdistance=0.7, textprops={'fontsize': 10})
    #plt.legend(patches, , loc="best")

    plt.tight_layout()
    if filename is not None: 
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".eps")

    return percentages

if __name__ == "__main__":
    models = pickle.load(open("results/cv5/chemistry-4.p", "rb"))
    # models_2 = pickle.load(open("results/full-2/medicine.p", "rb"))
    # models_3 = pickle.load(open("results/full-2/economics.p", "rb"))
    # models_4 = pickle.load(open("results/full-2/chemistry.p", "rb"))
    # models_5 = pickle.load(open("results/full-2/cs.p", "rb"))

    # for key in models:
    #      models[key].update(models_2[key])
    #      models[key].update(models_3[key])
    #      models[key].update(models_4[key])
    #      models[key].update(models_5[key])

    # set len_included=False for authorship analysis
    percent = make_pie_chart(get_winners(models, p_vals=False, include_null=False, len_included=True), include_null=False, filename="chem-fold-4")
    assert False
    get_pairwise_differences(models, include_null=False)
    #make_pie_chart(models)