import math
import pdb
from typing import List, Callable
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth

from src.models.predict import make_rank_on_1NN, get_sim, make_rank_on_exemplar, make_rank_on_prototype, rank_on_progenitor

MAX_CLUSTERS = 10


class CRP:
    def __init__(self, alpha: float, starting_points: List, likelihood_type: str, likelihood_hyperparam: float, first_cluster_size: int = 2, use_pca: bool = False, clustering_type: str = "kmeans") -> None:
        self.alpha = alpha
        self.first_cluster_size = first_cluster_size
        self.all_vecs = [np.array(point) for point in starting_points]
        self.clusters = [self.all_vecs]
        self.prev_epoch_clusters = None
        self.current_cluster_num = 1
        self.vecs_added = len(self.all_vecs)
        self.prev_vecs_added = 0
        self.timestep = 0
        self.rank_on_1NN = make_rank_on_1NN(1)
        self.use_pca = use_pca
        self.clustering_type = clustering_type
        self.likelihood_type = likelihood_type
        self.likelihood_hyperparam = likelihood_hyperparam

        if self.likelihood_type == "exemplar":
            self.ranking_function = make_rank_on_exemplar(s=self.likelihood_hyperparam)
        elif self.likelihood_type == "prototype":
            self.ranking_function = make_rank_on_prototype(s=self.likelihood_hyperparam)
        elif self.likelihood_type == "progenitor":
            self.ranking_function = rank_on_progenitor(starting_points, s=self.likelihood_hyperparam)
        elif self.likelihood_type == "local":
            self.ranking_function = make_rank_on_1NN(s=self.likelihood_hyperparam)

    def update_clusters(self, emerged: List, num_emerged: int) -> None:
        t0 = time()
        self.timestep += 1
        print("timestep: ", self.timestep)
        #print("emerged number: ", len(emerged))
        self.prev_vecs_added = self.vecs_added
        self.vecs_added += num_emerged
        #print("curr length: ", len(self.all_vecs))
        self.all_vecs.extend([np.array(item) for item in emerged])
        #print("length after: ", len(self.all_vecs))
        new_clusters = []
        best_sil_score = -float("inf")
        best_clusters = None

        if self.vecs_added <= self.first_cluster_size:
            self.clusters[0].extend([np.array(item) for item in emerged])
            return 

        if self.use_pca:
            pca = PCA(n_components=2)
            all_vecs = pca.fit_transform(self.all_vecs)
        else:
            all_vecs = self.all_vecs

        if self.clustering_type == "kmeans":
            for i in range(2, min(MAX_CLUSTERS + 1, self.vecs_added)):
                #print("clusters: ", i)
                #if self.timestep == 2 and i == 3:
                    #pdb.set_trace()
                Kmeans = KMeans(n_clusters=i)
                Kmeans.fit(all_vecs)
                labels = Kmeans.labels_
                proposed_clusters = {i: np.where(labels == i)[0] for i in range(Kmeans.n_clusters)}
                sil_score = silhouette_score(np.array(self.all_vecs), labels, metric="euclidean")
                #print(f"i:{i}, silhouette:{sil_score}")
                if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_clusters = proposed_clusters
                    self.current_cluster_num = i
        else:
            bandwidth = estimate_bandwidth(all_vecs)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
            ms.fit(all_vecs)
            labels = ms.labels_
            #best_clusters = {i: np.where(labels == i) for i in range()}

        self.prev_epoch_clusters = self.clusters
        if best_clusters is None:
            pdb.set_trace()
        for clust_num in best_clusters:
            best_vecs = [self.all_vecs[i] for i in best_clusters[clust_num]]
            new_clusters.append(best_vecs)

        t1 = time()
        #print("time: ", t1 - t0)
        self.clusters = new_clusters

    def rank_on_clusters_custom(self, unemerged: List) -> List:
        emerged = self.all_vecs
        if self.vecs_added < self.first_cluster_size:
            return self.ranking_function(emerged, unemerged, keep_sim=True)

        p_c_i_per_vec = {tuple(vec): -float("inf") for vec in unemerged}

        for i, cluster in enumerate(self.clusters):
            # compute p(i | c)p(c) for each cluster according to the likelihood fn
            p_c = math.log(len(cluster) / len(self.all_vecs), 2)
            pred_and_sim = self.ranking_function(cluster, unemerged, keep_sim=True)
            vecs, sims = zip(*pred_and_sim)
            total_sim = sum(sims)
            p_i_c_cluster = {}
            p_c_i_cluster = {}

            for vec, sim in zip(vecs, sims):
                p_i_c_cluster[vec] = sim/total_sim
                p_c_i_cluster[vec] = p_i_c_cluster[vec] + p_c

                # compute chance of forming a new cluster based on alpha
                p_i_c_new = 0
                p_c_new = math.log(self.alpha / (len(self.all_vecs) + self.alpha), 2)
                p_c_i_new = p_i_c_new + p_c_new
                if p_c_i_new > p_c_i_cluster[vec]:
                    p_c_i_cluster[vec] = p_c_i_new

                if p_c_i_cluster[vec] > p_c_i_per_vec[vec]:
                    p_c_i_per_vec[vec] = p_c_i_cluster[vec]

        # return candidates ranked by p_c_i
        predicted_rank_and_sim = [list(item) for item in sorted(p_c_i_per_vec.items(), key=lambda x: x[1], reverse=True)]

        return predicted_rank_and_sim

    def visualize(self, out_filename: str) -> None:
        for cluster in self.clusters:
            cluster_arr = np.array(cluster)
            plt.scatter(cluster_arr[:, 0], cluster_arr[:, 1])

        plt.savefig(out_filename)

def get_dist(vec_1: List, vec_2: List) -> float:
    return -get_sim(vec_1, vec_2)

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


if __name__ == "__main__":
    num_samples = 200
    cov1 = np.array([[2., 0.], [0., 1.]]) * 2
    cov2 = np.array([[1., 0.], [0., 1.]]) * 2
    cov3 = np.array([[3., 0.], [0., 1.]]) * 2
    mean_1 = [2., 3.]
    mean_2 = [5, 5.]
    mean_3 = [-1, -5]

    x_class1 = np.random.multivariate_normal(mean_1, cov1, num_samples // 3)
    x_class2 = np.random.multivariate_normal(mean_2, cov2, num_samples // 3)
    x_class3 = np.random.multivariate_normal(mean_3, cov3, num_samples // 3)

    data_full = np.row_stack([x_class1, x_class2, x_class3])
    np.random.shuffle(data_full)

    plt.scatter(x_class1[:, 0], x_class1[:, 1], marker='x') 
    plt.scatter(x_class2[:, 0], x_class2[:, 1], marker='o') 
    plt.savefig("scatter.png")
    plt.gcf().clear()

    print(data_full[0])

    crp = CRP(alpha=0.01, starting_point=data_full[0])
    rank = crp.rank_on_clusters(data_full[1:])
    crp.update_clusters(data_full[1:-10], len(data_full[1:-10]))
    predicted_rank = crp.rank_on_clusters(data_full[-10:])
    for i, vec in enumerate(predicted_rank):
        plt.text(vec[0], vec[1], i)
    print(predicted_rank)
    print("len", len(crp.clusters))
    crp.visualize("test_crp.png")

