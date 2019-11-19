import math
import pdb
from typing import List, Callable

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans

from .predict import rank_on_1NN


class CRP:
    def __init__(self, alpha: float, starting_point: List, first_cluster_size: int = 3) -> None:
        self.alpha = alpha
        self.first_cluster_size = first_cluster_size
        self.all_vecs = [starting_point]
        self.clusters = [[starting_point]]
        self.current_cluster_num = 1
        self.vecs_added = 1

    def update_clusters(self, emerged: List, num_emerged: int) -> None:
        self.vecs_added += num_emerged
        self.all_vecs.extend(emerged)
        cluster_proposals = []
        
        if self.current_cluster_num + num_emerged <= self.first_cluster_size:
            self.clusters[0].extend(emerged)
            return 

        for i in range(2, num_emerged + 1):
            Kmeans = KMeans(n_clusters=self.current_cluster_num + i)
            Kmeans.fit(self.all_vecs)
            labels = Kmeans.labels_
            proposed_clusters = {i: np.where(labels == i)[0] for i in range(Kmeans.n_clusters)}
            print(Kmeans.inertia_)
            cluster_proposals.append((Kmeans.inertia_, proposed_clusters))

        print(cluster_proposals)
        assert False
            
    def _update_clusters(self, emerged: List, num_emerged: int) -> None:
        if self.current_cluster_num + num_emerged <= self.first_cluster_size:
            self.clusters[0].extend(emerged)
            return
       
        assigned_clusters = {tuple(vec): -1 for vec in emerged}
        for vec in emerged: 
            attraction_by_cluster = {}
            for i in range(len(self.clusters)):
                attr = find_cluster_attraction(self.clusters[i], vec)
                attraction_by_cluster[i] = attr
            assigned_clusters[tuple(vec)] = max(attraction_by_cluster, key=attraction_by_cluster.get)
        
        print(assigned_clusters)
        for vec in assigned_clusters:
            self.clusters[assigned_clusters[vec]].append(list(vec))
         
    def rank_on_clusters(self, emerged: List, unemerged: List) -> List:
        if self.vecs_added < self.first_cluster_size:
            return rank_on_1NN(emerged, unemerged)

        all_vecs = emerged + unemerged
        p_c_i_per_vec = {}
        #pdb.set_trace()
        for candidate_vec in unemerged:
            max_c = 0
            for cluster in self.clusters:
                # compute P(i | c)p(c) for each cluster
                p_c = math.log(len(cluster) / len(all_vecs), 2)
                p_i_c = find_cluster_attraction(cluster, candidate_vec) # is this log base 2? 
                p_c_i = p_i_c + p_c
                max_c = p_c_i if p_c_i > max_c else max_c
                    
            # compute chance of forming a new cluster based on alpha
            p_i_c = 0
            p_c = math.log(self.alpha / (len(self.clusters) + self.alpha), 2)
            p_c_i = p_i_c + p_c
            max_c = p_c_i if p_c_i > max_c else max_c

            p_c_i_per_vec[tuple(candidate_vec)] = max_c

        # return candidates ranked by p_c_i
        predicted_rank = [list(item[0]) for item in sorted(p_c_i_per_vec.items(), key=lambda x: x[1])]
        
        return predicted_rank

    def calc_P_c_i(self, all_vecs: List, unemerged: List):
        p_c_i_per_vec = {}
        #pdb.set_trace()
        for candidate_vec in unemerged:
            max_c = 0
            for cluster in self.clusters:
                # compute P(i | c)p(c) for each cluster
                p_c = math.log(len(cluster) / len(all_vecs), 2)
                p_i_c = find_cluster_attraction(cluster, candidate_vec) # is this log base 2? 
                p_c_i = p_i_c + p_c
                max_c = p_c_i if p_c_i > max_c else max_c
                    
            # compute chance of forming a new cluster based on alpha
            p_i_c = 0
            p_c = math.log(self.alpha / (len(self.clusters) + self.alpha), 2)
            p_c_i = p_i_c + p_c
            max_c = p_c_i if p_c_i > max_c else max_c

            p_c_i_per_vec[tuple(candidate_vec)] = max_c

def create_crp(clusters: List, alpha: float, first_cluster_size: int = 3) -> Callable:
    def rank_on_clusters(emerged: List, unemerged: List) -> List:
        pdb.set_trace()
        if max(clusters, key=len) < first_cluster_size:
            return rank_on_1NN(emerged, unemerged)

        all_vecs = emerged + unemerged 
        p_c_i_per_vec = {}
        #pdb.set_trace()
        for candidate_vec in unemerged:
            max_c = 0
            for cluster in clusters:
                # compute P(i | c)p(c) for each cluster
                p_c = math.log(len(cluster) / len(all_vecs), 2)
                p_i_c = find_cluster_attraction(cluster, candidate_vec) # is this log base 2? 
                p_c_i = p_i_c + p_c
                max_c = p_c_i if p_c_i > max_c else max_c
                    
            # compute chance of forming a new cluster based on alpha
            p_i_c = 0
            p_c = math.log(alpha / (len(clusters) + alpha), 2)
            p_c_i = p_i_c + p_c
            max_c = p_c_i if p_c_i > max_c else max_c

            p_c_i_per_vec[tuple(candidate_vec)] = max_c

        # return candidates ranked by p_c_i
        predicted_rank = [list(item[0]) for item in sorted(p_c_i_per_vec.items(), key=lambda x: x[1])]
        
        return predicted_rank
        
    return rank_on_clusters


def find_cluster_attraction(cluster: List, vec: np.ndarray) -> float:
    kernel = KernelDensity(kernel="gaussian").fit(cluster)
    prob = kernel.score(np.asarray(vec).reshape(1, -1))
    return prob