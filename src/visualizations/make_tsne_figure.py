import csv
import random
from typing import Callable, List

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.manifold import TSNE
#from torch.utils.tensoPrboard import SummaryWriter
#from torchvision import dat
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
import pdb
from pprint import pprint

def select_subset(
    vecs_filepath: str,
    labels_filepath: str,
    out_vec_filepath: str,
    out_labels_filepath: str,
    filter: Callable[[str], bool],
    threshold: float = 0.5,
    filter_on: bool = "title",
) -> None:
    with open(vecs_filepath, "r") as vecs_file, open(
        labels_filepath, "r"
    ) as labels_file, open(out_vec_filepath, "w") as out_vecs_file, open(
        out_labels_filepath, "w"
    ) as out_labels_file:
        vecs_reader = csv.reader(vecs_file)
        labels_reader = csv.reader(labels_file)
        vecs_writer = csv.writer(out_vecs_file)
        labels_writer = csv.writer(out_labels_file)

        for vec_row, label_row in zip(vecs_reader, labels_reader):
            year, title = label_row
            criterion_met = (filter_on == "title" and filter(title)) or (
                filter_on == "year" and filter(year)
            )
            randomly_chosen = random.random() <= threshold
            if criterion_met and randomly_chosen:
                vecs_writer.writerow(vec_row)
                labels_writer.writerow(label_row)

def get_tsne_from_json(filepath: str) -> np.ndarray:
    with open(filepath, "r") as f:
        data = json.load(f)
        pprint(data) 

def plot_tsne(
    vecs: np.ndarray, years: np.ndarray, labels: np.ndarray, out_filename: str = "papers_tsne", three_d: bool = False, field: str = "cs"
) -> None:
    # Hyperparams determined through some trial and error in embedding projector...
    n_components = 3 if three_d else 2
    perplexity = 1 #1 for cs, 10 for phys
    name = "hinton"

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=100,
        n_iter=100000,
        n_iter_without_progress=10000
    )

    selected = {
        "cs": {
            "hinton": {
                "using relaxation find puppet": "1", #1976
                "demonstration effect structural description mental imagery": "2", #1979
                "dynamic routing capsule": "6.1", #2017
                "dropout simple way prevent neural network overfitting": "5", #2014
                "analyzing improving representation soft nearest neighbor loss": "7", #2019
                "deep belief net": "6.2", #2017
                "learning algorithm boltzmann machine": "3", #1985
                "spiking boltzmann machine": "4" #1999
            }
        },
       "physics": {
           "bohr": {
            "The Peaceful Uses of Atomic Energy": "The Peaceful Uses of Atomic Energy\n1955",
           "Nuclear Photo-effects": "Nuclear Photo-effects\n1938",
           "Conservation Laws in Quantum Theory": "Conservation Laws in Quantum Theory\n1936",
           "Deuteron-Induced Fission": "Deuteron-induced Fission\n1941",
           "Velocity-range relation for fission fragments": "Velocity-range relation for fission fragments\n1940",
           "Natural Philosophy and Human Cultures": "Natural Philosophy and Human Cultures\n1939"
           },
           "wilczek": {
               "Truncated dynamics, ring molecules, and mechanical time crystals": "Truncated dynamics, ring molecules, and mechanical time crystals\n2019",
               "Efficient Quantum Algorithm for a Class of 2-SAT Problems": "Efficient Quantum Algorithm for a Class of 2-SAT Problems\n2018",
               "Thermal Decay of the Cosmological Constant into Black Holes": "Thermal Decay of the Cosmological Constant into Black Holes\n2003",
               "Ultraviolet Behavior of Non-Abelian Gauge Theories": "Ultraviolet Behavior of Non-Abelian Gauge Theories\n1973",
               "Orientation of the weak interaction with respect to the strong interaction": "Orientation of the weak interaction with respect to the strong interaction\n1977",
               "Limitations on the statistical description of black holes": "Limitations on the statistical description of black holes\n1991",
               "Quantum hair on black holes": "Quantum hair on black holes\n1992",

           },
           "strickland": {
               "Compression of amplified chirped optical pulses": "Compression of amplified chirped optical pulses\n1985",
               "Laser ionization of noble gases by Coulomb-barrier suppression": "Laser ionization of noble gases by Coulomb-barrier suppression\n1991",
               "Dual-wavelength chirped-pulse amplification system.": "Dual-wavelength chirped-pulse amplification system\n2000",
               "Education program for photonics professionals": "Education program for photonics professionals\n2003",
               "Effect of laser beam induced nonlinearity on the performance of crystalline lens": "Effect of laser beam induced nonlinearity on the performance of crystalline lens\n2007",
               "Effect of plasma generation on the performance of the crystalline lens": "Effect of plasma generation on the performance of the crystalline lens\n2013",
               "Nonlinear laser pulse response in a crystalline lens.": "Nonlinear laser pulse response in a crystalline lens\n2016",
               "Temporal and Spectral Measurement of Red-Shifted Spectrum in Multi-Frequency Raman Generation": "Temporal and Spectral Measurement of Red-Shifted Spectrum in Multi-Frequency Raman Generation\n2018"
           },
           "feynman": {
               "The scattering of cosmic rays by the stars of a galaxy": "1", #1939
               "The principle of least action in quantum mechanics": "2", #1942
               "Atomic theory of liquid helium near absolute zero": "3", #1953
               "Slow electrons in a polar crystal": "6.1", #1962
               "Dispersion of the neutron emission in U-235 fission☆": "4", #1956
               "Superfluidity and Superconductivity": "5", #1957
               "Mobility of Slow Electrons in a Polar Crystal": "6.2", #1962
               "The behavior of hadron collisions at extreme energies": "7", #1969
               "Simulating physics with computers": "8", #1982
               "Quantum Mechanical Computers": "9" #1986
           }
       },
       "chemistry": {
           "goodenough": {
               "Lithium anode stable in air for low-cost fabrication of a dendrite-free lithium battery": "8", #2019
               "A high-performance all-metallocene-based, non-aqueous redox flow battery": "7", #2017 
               "Perspective on Engineering Transition-Metal Oxides": "6", #2014
               "Solid Oxide Fuel Cell Technology: Principles, Performance and Operations": "5", #2009
               "Supercapacitor Behavior with KCl Electrolyte": "4", #1999
               "Oxide-ion conduction in Ba2In2O5 and Ba3In2MO8 (M=Ce, Hf, or Zr)": "3", #1990
               "Photoelectrochemistry of nickel(II) oxide": "2", #1981
               "A Theory of Domain Creation and Coercive Force in Polycrystalline Ferromagnetics": "1" #1954
           },
           "pauling": {
               "The calculation of matrix elements for Lewis electronic structures of molecules": "The calculation of matrix elements for Lewis electronic structures of molecules\n1933",
               "Atomic Radii and Interatomic Distances in Metals": "Atomic Radii and Interatomic Distances in Metals\n1947",
               "Nature of Forces between Large Molecules of Biological Interest": "Nature of Forces between Large Molecules of Biological Interest\n1948",
               "Resonance in the Hydrogen Molecule": "Resonance in the Hydrogen Molecule\n1952",
               "The Dependence of Bond Energy on Bond Length": "The Dependence of Bond Energy on Bond Length\n1954",
               "Ascorbic acid and cancer: a review.": "Ascorbic acid and cancer: a review.\n1979"
           }
       },
       "medicine": {
           "krebs": {
               "The citric acid cycle.": "The citric acid cycle\n1940",
               "Body size and tissue respiration.": "Body size and tissue respiration\n1950",
               "Gluconeogenesis in the perfused rat liver": "Gluconeogenesis in the perfused rat liver\n1966",
               "Carbohydrate metabolism of the perfused rat liver.": "Carbohydrate metabolism of the perfused rat liver\n1967",
               "Inhibition of hepatic gluconeogenesis by ethanol.": "Inhibition of hepatic gluconeogenesis by ethanol\n1969",
               "Ketone-body utilization by adult and suckling rat brain in vivo.": "Ketone-body utilization by adult and suckling rat brain in vivo\n1971",
               "Metabolism and excretion of normorphine in dogs.": "Metabolism and excretion of normorphine in dogs\n1978",
               "Utilization of energy-providing substrates in the isolated working rat heart.": "Utilization of energy-providing substrates in the isolated working rat heart\n1980"
           },
           "yamanaka": {
               "Towards Precision Medicine With Human iPSCs for Cardiac Channelopathies.": "Towards Precision Medicine With Human iPSCs for Cardiac Channelopathies.\n2019",
               "Induced pluripotent stem cell technology: a decade of progress": "Induced pluripotent stem cell technology: a decade of progress\n2017",
               "A novel efficient feeder-free culture system for the derivation of human induced pluripotent stem cells": "A novel efficient feeder-free culture system for the derivation of human induced pluripotent stem cells\n2014",
               "A chemical probe that labels human pluripotent stem cells.": "A chemical probe that labels human pluripotent stem cells.\n2014",
               "Steps toward safe cell therapy using induced pluripotent stem cells.": "Steps toward safe cell therapy using induced pluripotent stem cells.\n2013",
               "Generation of Human Melanocytes from Induced Pluripotent Stem Cells": "Generation of Human Melanocytes from Induced Pluripotent Stem Cells\n2011",
               "Reactivation of the Paternal X Chromosome in Early Mouse Embryos": "Reactivation of the Paternal X Chromosome in Early Mouse Embryos\n2004",
               "Role of hypomagnesemia in chronic cyclosporine nephropathy.": "Role of hypomagnesemia in chronic cyclosporine nephropathy.\n2002"
           }
       },
       "economics": {
           "kahneman": {
               "Advances in prospect theory: Cumulative representation of uncertainty": "5", #1992
               "Prospect theory: An analysis of decision under risk Econometrica 47": "4", #1979
               "Causal Thinking in Judgment under Uncertainty": "3", #1977
               "Thinking, Fast and Slow": "10", #2011
               "High income improves evaluation of life but not emotional well-being": "9", #2010
               "National Time Accounting: The Currency of Life": "8", #2008
               "Would You Be Happier If You Were Richer? A Focusing Illusion": "7", #2006
               "A perspective on judgment and choice: mapping bounded rationality.": "6", #2003
               "VISUAL RESOLUTION AND CONTOUR INTERACTION.": "1", #1963
                "Pupil Diameter and Load on Memory": "2" #1966
           }
       }
    }
    
    tsne_res = tsne.fit_transform(vecs)

    years = years.astype(np.int64)
    if three_d:
        seaborn.set(style="darkgrid")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(tsne_res[:, 0], tsne_res[:, 1], tsne_res[:, 2])

        for i in range(0, tsne_res.shape[0]):
            if labels[i] in selected[field][name]:
                ax.text(tsne_res[i, 0], tsne_res[i, 1], tsne_res[i, 2], selected[field][name][labels[i]], size="small")

    else:
        seaborn.scatterplot(
            tsne_res[:, 0],
            tsne_res[:, 1],
            hue=years,
            hue_norm=matplotlib.colors.Normalize(vmin=min(years), vmax=max(years)),
            palette="plasma"
        )
        ax = plt.gca()
        for i in range(0, tsne_res.shape[0]):
            if labels[i] in selected[field][name]:
                ax.text(tsne_res[i, 0], tsne_res[i, 1], selected[field][name][labels[i]], size="small")

    plt.axis("off")
    #plt.tight_layout()
    plt.savefig(out_filename + ".png")
    plt.savefig(out_filename + ".eps")
    plt.show()

    return tsne_res

if __name__ == "__main__":
    #get_tsne_from_json("./results/tsne/tsne-res.txt")
    #assert False
    field = "cs"
    paths = {
        "cs": {
            "data": "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-vecs-final.tsv",
            "years": "./data/turing_winners/abstracts/geoff/Geoff-Hinton-abstract-labels-final.tsv"
        },
        "physics": {
            "data": "./data/nobel_winners/physics/feynman/vecs.tsv",
            "years": "./data/nobel_winners/physics/feynman/labels.tsv"
        },
        "chemistry": {
            "data": "./data/nobel_winners/chemistry/goodenough/vecs.tsv",
            "years": "./data/nobel_winners/chemistry/goodenough/labels.tsv"
        },
        "medicine": {
            "data": "./data/nobel_winners/medicine/krebs/vecs.tsv",
            "years": "./data/nobel_winners/medicine/krebs/labels.tsv"
        },
        "economics": {
            "data": "./data/nobel_winners/economics/kahneman/vecs.tsv",
            "years": "./data/nobel_winners/economics/kahneman/labels.tsv"
        }
    }
    delimiter = ","
    """select_subset(
        "hinton_paper_vectors.csv",
        "hinton_papers.csv",
        "hinton_vecs_selected.csv",
        "hinton_titles_selected.csv",
        lambda s: len(s.split(" ")) <= 7)"""
    data = np.genfromtxt(paths[field]["data"], delimiter="\t")

    years = np.genfromtxt(
        paths[field]["years"], delimiter="\t", skip_header=1, usecols=(0)
    )
    labels = np.genfromtxt(
        paths[field]["years"], dtype="str", delimiter="\t", skip_header=1, usecols=(1)
    )
    print(data, years, labels)
    tsne_res = plot_tsne(data, years, labels, three_d=False, out_filename="hinton", field=field)

