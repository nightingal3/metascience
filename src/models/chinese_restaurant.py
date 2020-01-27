import math
import pdb
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans

from src.models.predict import rank_on_1NN


class CRP:
    def __init__(self, alpha: float, starting_point: List, first_cluster_size: int = 3) -> None:
        self.alpha = alpha
        self.first_cluster_size = first_cluster_size
        self.all_vecs = [starting_point]
        self.clusters = [[starting_point]]
        self.prev_epoch_clusters = None
        self.current_cluster_num = 1
        self.vecs_added = 1
        self.prev_vecs_added = 0
        self.timestep = 0

    def update_clusters(self, emerged: List, num_emerged: int) -> None:
        self.timestep += 1
        print("timestep: ", self.timestep)
        self.prev_vecs_added = self.vecs_added
        self.vecs_added += num_emerged
        self.all_vecs.extend(emerged)
        cluster_proposals = []

        if self.vecs_added < self.first_cluster_size:
            self.clusters[0].extend(emerged)
            return 

        for i in range(2, self.vecs_added):
            Kmeans = KMeans(n_clusters=self.current_cluster_num + i)
            Kmeans.fit(self.all_vecs)
            labels = Kmeans.labels_
            proposed_clusters = {i: np.where(labels == i)[0] for i in range(Kmeans.n_clusters)}
            print(Kmeans.inertia_)
            cluster_proposals.append((Kmeans.inertia_, proposed_clusters))

        print("cluster proposals: ", cluster_proposals)
        
            
    def _update_clusters(self, emerged: List, num_emerged: int) -> None:
        self.timestep += 1
        self.prev_vecs_added = self.vecs_added
        self.vecs_added += num_emerged
        if self.vecs_added < self.first_cluster_size:
            self.clusters[0].extend(emerged)
            return
        pdb.set_trace()
        assigned_clusters = {tuple(vec): -1 for vec in emerged}
        for i, vec in enumerate(emerged): 
            attraction_by_cluster = {}
            for i in range(len(self.clusters)): # Find attraction to old clusters 
                attr = find_cluster_attraction(self.clusters[i], vec)
                attraction_by_cluster[i] = attr
            assigned_clusters[tuple(vec)] = max(attraction_by_cluster, key=attraction_by_cluster.get)

            # Attraction to completely new cluster (by itself)
            new_cluster_attr = self.alpha / (self.prev_vecs_added + i + self.alpha - 1)

            # Attraction to new cluster (with previous data from this epoch)
            # ...

            
        print("assigned clusters: ", assigned_clusters)
        for vec in assigned_clusters:
            self.clusters[assigned_clusters[vec]].append(list(vec))
        

    def rank_on_clusters(self, emerged: List, unemerged: List) -> List:
        if self.vecs_added < self.first_cluster_size:
            return rank_on_1NN(emerged, unemerged)
        print(emerged)
        all_vecs = emerged.extend(unemerged)
        p_c_i_per_vec = {}
        #pdb.set_trace()
        for candidate_vec in unemerged:
            max_c = -float("inf")
            for cluster in self.clusters:
                # compute P(i | c)p(c) for each cluster
                p_c = math.log(len(cluster) / len(self.all_vecs), 2)
                p_i_c = find_cluster_attraction(cluster, candidate_vec) # is this log base 2? 
                p_c_i = p_i_c + p_c
                max_c = p_c_i if p_c_i > max_c else max_c
                    
            # compute chance of forming a new cluster based on alpha
            p_i_c_new = 0
            p_c_new = math.log(self.alpha / (len(self.clusters) + self.alpha), 2)
            p_c_i_new = p_i_c_new + p_c_new
            max_c = p_c_i_new if p_c_i_new > max_c else max_c

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



if __name__ == "__main__":
    num_samples = 200
    cov1 = np.array([[1., 0.], [0., 1.]]) * 2
    cov2 = np.array([[1., 0.], [0., 1.]]) * 2
    cov3 = np.array([[1., 0.], [0., 1.]]) * 2
    mean_1 = [-5., -5.]
    mean_2 = [5, 5.]

    x_class1 = np.random.multivariate_normal(mean_1, cov1, num_samples // 2)
    x_class2 = np.random.multivariate_normal(mean_2, cov2, num_samples // 2)
   
    data_full = np.row_stack([x_class1, x_class2])
    np.random.shuffle(data_full)

    plt.scatter(x_class1[:, 0], x_class1[:, 1], marker='x') 
    plt.scatter(x_class2[:, 0], x_class2[:, 1], marker='x') 

    print(data_full[0])

    crp = CRP(alpha=0.01, starting_point=data_full[0])
    rank = crp.rank_on_clusters(crp.all_vecs, data_full[1:])
    print(rank[:3])
    crp.update_clusters([data_full[1]], 1)
    crp.update_clusters([data_full[2]], 1)
    crp.update_clusters([data_full[3]], 1)
    print(crp.rank_on_clusters(crp.all_vecs, data_full[4:]))

