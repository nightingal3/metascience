import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from collections import OrderedDict

def get_winners(model_dict: dict, p_vals=True) -> dict:
    scientists_to_models = {k: (-float("inf"), None) for k in model_dict['1NN']}
    for model in model_dict:
        print("MODEL: ", model)
        for scientist in model_dict[model]:
            print("SCIENTIST: ", scientist)
            if p_vals and model_dict[model][scientist][1] > 0.05 * len(model_dict[model]):
                continue
            if p_vals:
                val = model_dict[model][scientist][0]
            else:
                val = model_dict[model][scientist]
            print(val)
            if val > scientists_to_models[scientist][0]:
                scientists_to_models[scientist] = (val, model)
    #scientists_to_models = {k: v if v[0] > 0 else (0, "Null") for k, v in scientists_to_models.items()}

    return scientists_to_models

def make_pie_chart(scientists_to_models) -> None:
    percentages = {}
    for scientist in scientists_to_models:
        if scientists_to_models[scientist][1] not in percentages:
            percentages[scientists_to_models[scientist][1]] = 1
        else:
            percentages[scientists_to_models[scientist][1]] += 1
    order = {
        "1NN": 0,
        "2NN": 1,
        "3NN": 2,
        "4NN": 3,
        "5NN": 4, 
        "Progenitor": 5,
        "Prototype": 6,
        "Exemplar": 7,
        "Local": 8,
        "Null": 9
    }
    percentages = {k: v / len(scientists_to_models) for k, v in percentages.items()}
    percent_sorted = sorted(list(percentages.items()), key=lambda x: order[x[0]])
    plt.pie([i[1] for i in percent_sorted], labels=[i[0] for i in percent_sorted], autopct='%1.1f%%', colors=cm.Set3.colors, pctdistance=0.7, textprops={'fontsize': 16})
    plt.tight_layout()
    plt.savefig("best-models-overall-ll.png")
    plt.savefig("best-models-overall-ll.eps")


if __name__ == "__main__":
    models = get_winners(pickle.load(open("model-LL.p", "rb")), p_vals=False)
    make_pie_chart(models)