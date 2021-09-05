import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import cycle, islice
import pdb

order = OrderedDict({
    "kNN": 0,
    #"2NN": 1,
    #"3NN": 2,
    #"4NN": 3,
    #"5NN": 4, 
    "exemplar": 5,
    #"Exemplar (s=0.001)": 6,
    #"Exemplar (s=0.1)": 7,
    #"Exemplar (s=1)": 8,
    #"Exemplar":9,
    "progenitor": 10,
    "prototype": 11,
    #"Exemplar (s=1)": 8,
    "local": 12,
    "Null": 13
})

colours = dict(zip(order.keys(), plt.cm.Set3.colors[:len(order)]))

# from https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        color_cycle = list(colours.values())
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=color_cycle,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    print((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    plt.xlim(-0.5, 4.8)
    axe.set_xticklabels(df.index, rotation = 0)
    #axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    #plt.margins(-0.4, 0.1)
    plt.tight_layout()
    plt.savefig("authorship_rand.png")
    plt.savefig("authorship_rand.eps")

    return axe

if __name__ == "__main__":
    df1 = pd.DataFrame(np.array([
                        [0.328, 0.015, 0.045, 0.582, 0.03, 0],
                       [0.299, 0.015, 0.119, 0.373, 0.03, 0.164],
                       [0.338, 0.015, 0.138, 0.354, 0.015, 0.138],
                       [0.292, 0.046, 0.138, 0.385, 0.015, 0.123],
                       [0.245, 0.094, 0.245, 0.358, 0, 0.057] 
                       ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])
    df1_rand = pd.DataFrame(np.array([
                        [0.159, 0, 0.377, 0.406, 0, 0.058],
                        [0.111, 0, 0.333, 0.4, 0, 0.156],
                       [0.156, 0.062, 0.312, 0.375, 0.031, 0.062],
                       [0.125, 0, 0.375, 0.458, 0, 0.042],
                       [0.083, 0.083, 0.333, 0.333, 0, 0.167] 
                       ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])

    df2 = pd.DataFrame(np.array([
                    [0.075, 0.008, 0.308, 0.583, 0, 0.025],
                    [0.056, 0.047, 0.227, 0.605, 0.02, 0.045],
                    [0.078, 0.029, 0.206, 0.618, 0.02, 0.049],
                    [0.093, 0.001, 0.206, 0.608, 0.031, 0.052],
                    [0.133, 0.033, 0.2, 0.517, 0.033, 0.083]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])
    df2_rand = pd.DataFrame(np.array([
                    [0.1, 0, 0.308, 0.567, 0, 0.025],
                    [0.119, 0.017, 0.373, 0.424, 0, 0.068],
                    [0.077, 0.051, 0.359, 0.41, 0.077, 0.026],
                    [0.192, 0.038, 0.192, 0.423, 0.115, 0.038],
                    [0.25, 0, 0.333, 0.417, 0, 0]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])

    df3 = pd.DataFrame(np.array([
                    [0.162, 0.014, 0.243, 0.527, 0.014, 0.041],
                    [0.157, 0, 0.2, 0.557, 0.057, 0.029],
                    [0.143, 0, 0.229, 0.529, 0.071, 0.029],
                    [0.186, 0, 0.186, 0.543, 0.043, 0.043],
                    [0.121, 0.045, 0.212, 0.5, 0.015, 0.106]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])
    df3_rand = pd.DataFrame(np.array([
                    [0.135, 0, 0.297, 0.5, 0.014, 0.054],
                    [0.135, 0, 0.216, 0.514, 0.054, 0.081],
                    [0.147, 0.029, 0.265, 0.441, 0.059, 0.049],
                    [0.143, 0, 0.214, 0.464, 0.071, 0.107],
                    [0.214, 0, 0.429, 0.214, 0.071, 0.071]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])

    df4 = pd.DataFrame(np.array([
                    [0.026, 0.02, 0.225, 0.709, 0, 0.021],
                    [0.027, 0.014, 0.247, 0.658, 0.027, 0.027],
                    [0.063, 0.021, 0.261, 0.599, 0.035, 0.021],
                    [0.095, 0.036, 0.226, 0.547, 0.066, 0.029],
                    [0.106, 0.032, 0.245, 0.543, 0.032, 0.043]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])
    df4_rand = pd.DataFrame(np.array([
                    [0.073, 0, 0.338, 0.517, 0.013, 0.06],
                    [0.048, 0.024, 0.289, 0.482, 0.048, 0.108],
                    [0.17, 0.019, 0.264, 0.434, 0.038, 0.075],
                    [0.132, 0.026, 0.132, 0.553, 0.026, 0.132],
                    [0.176, 0.059, 0.235, 0.412, 0, 0.118]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])
    df5 = pd.DataFrame(np.array([
                    [0.077, 0.012, 0.405, 0.464, 0, 0.042],
                    [0.114, 0.03, 0.197, 0.591, 0.045, 0.023],
                    [0.107, 0.032, 0.216, 0.552, 0.064, 0.032],
                    [0.131, 0.033, 0.279, 0.484, 0.041, 0.033],
                    [0.11, 0.037, 0.244, 0.476, 0.049, 0.085]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])
    df5_rand = pd.DataFrame(np.array([
                    [0.131, 0.012, 0.244, 0.53, 0.018, 0.065],
                    [0.077, 0.038, 0.173, 0.577, 0.067, 0.067],
                    [0.122, 0.054, 0.203, 0.432, 0.068, 0.122],
                    [0.115, 0.038, 0.212, 0.404, 0.077, 0.154],
                    [0.143, 0.048, 0.19, 0.524, 0.048, 0.048]
                    ]),
                    index=["All", "First author", "3-author", "2-author", "1-author"],
                    columns=["kNN", "Prototype", "Progenitor", "Exemplar", "Local", "Null"])

    #plot_clustered_stacked([df1, df2, df3, df4, df5],["CS", "Chemistry", "Economics", "Medicine", "Physics"])
    plot_clustered_stacked([df5_rand, df2_rand, df4_rand, df3_rand, df1_rand], ["Physics random", "Chemistry random", "Medicine random", "Economics random", "CS random"])
