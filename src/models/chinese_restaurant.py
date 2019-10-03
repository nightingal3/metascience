from typing import List, Callable

import numpy as np
from scipy.stats import gaussian_kde

def create_crp(clusters: List, alpha: float, first_cluster_size: int = 3) -> Callable:
    def rank_on_clusters(emerged: List, unemerged: List) -> List:
        all_vecs = emerged + unemerged 
        p_c_i_per_vec = {}
        for candidate_vec in unemerged:
            max_c = 0
            for cluster in clusters:
                # compute P(i | c)p(c) for each cluster
                p_c = math.log(len(cluster) / len(all_vecs), 2)
                p_i_c = math.log(find_cluster_attraction(cluster, candidate_vec), 2)
                p_c_i = p_i_c + p_c
                max_c = p_c_i if p_c_i > max_c else max_c
                    
            # compute chance of forming a new cluster based on alpha
            p_i_c = 0
            p_c = math.log(alpha / (len(clusters) + alpha), 2)
            p_c_i = p_i_c + p_c
            max_c = p_c_i if p_c_i > max_c else max_c

            p_c_i_per_vec[tuple(candidate_vec)] = max_c

        # return candidates ranked by p_c_i
        predicted_rank = [item[0] for item in sorted(p_c_i_per_vec.items(), key=lambda x: x[1])]
        return predicted_rank
        
    return rank_on_clusters


def find_cluster_attraction(cluster: List, vec: np.ndarray) -> float:
    kernel = gaussian_kde(np.asarray(cluster))
    prob = kernel.evaluate(vec)
    return prob