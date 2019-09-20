
from functools import reduce
from typing import List, Callable

import numpy as np
from sklearn import preprocessing, neighbors, metrics

def rank_on_1NN(emerged: List, unemerged: List) -> List:
    closest = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sim = get_sim(emerged_vec, candidate_vec)
            if sim > closest[tuple(candidate_vec)]:
                closest[tuple(candidate_vec)] = sim[0][0]

    return [list(item[0]) for item in sorted([item for item in closest.items()], key=lambda x: x[1], reverse=True)]

# Rank on local is implemented by limiting the emerged list to points that emerged at the last timestep
def rank_on_prototype(emerged: List, unemerged: List) -> List:
    closest = {tuple(vec): 0 for vec in unemerged}
    
    proto_vec = get_prototype(emerged)
    for candidate_vec in unemerged:
        sim = get_sim(candidate_vec, proto_vec)
        closest[tuple(candidate_vec)] = sim[0][0]
    return [list(item[0]) for item in sorted([item for item in closest.items()], key=lambda x: x[1], reverse=True)]
    
def get_prototype(vecs: List) -> List:
    sum_vec = reduce(np.add, vecs)
    return np.divide(sum_vec, len(vecs))

def rank_on_progenitor(progenitor: tuple) -> Callable:
    def rank(emerged: List, unemerged: List) -> List:
        closest = {tuple(vec): 0 for vec in unemerged}
        for vec in unemerged:
            sim = get_sim(progenitor, vec)
            closest[tuple(vec)] = sim[0][0]

        return [list(item[0]) for item in sorted([item for item in closest.items()], key=lambda x: x[1], reverse=True)]

    return rank

def rank_on_exemplar(emerged: List, unemerged: List) -> List:
    sum_sim = {tuple(vec): 0 for vec in unemerged}

    for emerged_vec in emerged:
        for candidate_vec in unemerged:
            sum_sim[tuple(candidate_vec)] += get_sim(emerged_vec, candidate_vec)
    
    return [list(item[0]) for item in sorted([item for item in sum_sim.items()], key=lambda x: x[1], reverse=True)]

def get_sim(vec_1: List, vec_2: List) -> float:
    return metrics.pairwise.cosine_similarity(np.asarray(vec_1).reshape(1, -1), np.asarray(vec_2).reshape(1, -1))

def get_similarity_decreasing(lst: List) -> List:
    return [list(item[0]) for item in sorted([item for item in lst.items()], key=lambda x: x[1], reverse=True)]
    
if __name__ == "__main__":
    print("Hello world")