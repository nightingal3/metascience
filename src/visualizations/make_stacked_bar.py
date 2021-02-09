import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from collections import OrderedDict
from typing import List
import operator
from functools import reduce

from make_pie_chart import get_winners
import pdb

order = OrderedDict({
    "1NN": 0,
    "2NN": 1,
    "3NN": 2,
    "4NN": 3,
    "5NN": 4, 
    "Exemplar": 5,
    "Progenitor": 6,
    "Prototype": 7,
    #"Exemplar (s=1)": 8,
    "Local": 8
})

order_fields = OrderedDict({
    "CS": 0, 
    "Chemistry": 1,
    "Physics": 2,
    "Medicine": 3,
    "Economics": 4
})

colours = dict(zip(order.keys(), plt.cm.Set3.colors[:len(order)]))

def count_scientists_per_domain(**winners_data: dict) -> dict:
    winners = {}
    for name, domain_data in winners_data.items():
        count = [0] * len(order)

        for scientist in domain_data:
            print(domain_data[scientist])
            if domain_data[scientist][1] == None or domain_data[scientist][1][0] == "Null":
                continue
            if len(domain_data[scientist][1]) == 1:
                count[order[domain_data[scientist][1][0]]] += 1
            else:
                for mod in domain_data[scientist][1]:
                    if mod not in count:
                        count[order[mod]] = 1/len(domain_data[scientist][1])
                    else:
                        count[order[mod]] += 1/len(domain_data[scientist][1])
        winners[name] = count

    return winners

def _stack_bar_chart(**count_data: List) -> None:
    curr_height = np.zeros((1, len(order)))
    item_order = list(count_data.items())
    for name, model_count in item_order:
        curr_height = curr_height + np.asarray(model_count)
        print(curr_height)
        count_data[name] = curr_height[0]
    
    for name, model_count in item_order[::-1]:
        sns.barplot(x=list(range(len(order))), y=count_data[name], color=colours[name], label=name)

    plt.xticks(range(len(order)), order.keys(), rotation=45)
    plt.ylabel("Number of scientists", fontsize=14)
    plt.xlabel("Model", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test-bar-stack-1.png")
    plt.savefig("test-bar-stack-1.eps")


def stack_bar_chart(**fields: dict) -> None:
    x_vals = []
    field_names = []
    seen_models = set()
    for x_val, (name, scientists_to_models) in enumerate(fields.items()):
        x_vals.append(x_val)
        field_names.append(name)
        percentages = {}
        for scientist in scientists_to_models:
            if scientists_to_models[scientist][1] == None or scientists_to_models[scientist][1][0] == "Null":
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

        num_not_null = len([v for k, v in scientists_to_models.items() if v[1] and v[1][0] != "Null"])

        percentages = {k: v / num_not_null for k, v in percentages.items()}
        #print(sorted(list(percentages.items()), key=operator.itemgetter(1), reverse=True))
        percent_sorted = sorted(list(percentages.items()), key=lambda x: order[x[0]])
        y_vals = []
        labels = []
        curr_height = 0
        for model_name, percent in percent_sorted:
            curr_height += percent
            y_vals.append(curr_height)
            labels.append(model_name)

        print(y_vals)
        print(labels)
        
        for label, height in zip(labels[::-1], y_vals[::-1]):
            if x_val == 0 or label not in seen_models:
                plt.bar(x_val, height, label=label, color=colours[label])
            else:
                plt.bar(x_val, height, color=colours[label])
            
            seen_models.add(label)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(x_vals, field_names, rotation=45)
    plt.xlabel("Field", fontsize=14)
    plt.ylabel("Proportion of scientists", fontsize=14)
    plt.tight_layout()
    plt.savefig("stacked-bar-2.png")
    plt.savefig("stacked-bar-2.eps")
            

if __name__ == "__main__":
    models_cs = pickle.load(open("results/summary/cv-final/cs-2.p", "rb"))
    models_econ = pickle.load(open("results/summary/cv-final/economics-2.p", "rb"))
    models_med = pickle.load(open("results/summary/cv-final/medicine-2.p", "rb"))
    models_chem = pickle.load(open("results/summary/cv-final/chemistry-2.p", "rb"))
    models_phys = pickle.load(open("results/summary/cv-final/physics-2.p", "rb"))

    winners_cs = get_winners(models_cs, p_vals=False)
    winners_econ = get_winners(models_econ, p_vals=False)
    winners_med = get_winners(models_med, p_vals=False)
    winners_chem = get_winners(models_chem, p_vals=False)
    winners_phys = get_winners(models_phys, p_vals=False)


    #all_counts = count_scientists_per_domain(CS=winners_cs, Economics=winners_econ, Medicine=winners_med, Chemistry=winners_chem, Physics=winners_phys)
    stack_bar_chart(CS=winners_cs, Economics=winners_econ, Medicine=winners_med, Chemistry=winners_chem, Physics=winners_phys)
