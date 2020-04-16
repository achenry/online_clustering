   # else:
        #     # check if centroids should be reinitialised
        #     init = False
        #     centre_of_masses = [c.centre_of_mass for c in self.clusters]
        #
        # # check if any centre of masses are significantly close,
        # # if so merge them, reduce num clusters and reinitialise
        # com_distances = np.tril(pairwise_distances(centre_of_masses))
        # com_distances = com_distances[np.where(com_distances != 0)]
        # z = np.abs(stats.zscore(com_distances, axis=None))
        #
        # outlier_indices = np.concatenate(np.asarray(z > self.outlier_threshold).nonzero())
        # for k in outlier_indices:
        #     new_centroid = np.mean([c.centroid for c in self.clusters[k]], axis=0)
        #     self.clusters = np.delete(self.clusters, k)
        #     self.clusters = np.append(self.clusters,
        #                               [self.cluster_class(centroid=new_centroid)])
        #     init = True

        #     # check if any centroids and centre of masses are significantly far, if so split them, increase num clusters
        #     # and reinitialise
        #     centroid_com_distances = np.array([np.sum((c.centroid - c.centre_of_mass) ** 2) ** 0.5
        #                                        for c in self.clusters])
        #     centroid_com_distances = centroid_com_distances[np.where(centroid_com_distances != 0)]
        #     z = np.abs(stats.zscore(centroid_com_distances, axis=None))
        #     # todo should weights be normalised such that new data points can still move old and heavy clusters
        #
        #     outlier_indices = np.concatenate(np.asarray(z > self.outlier_threshold).nonzero())
        #     for k in outlier_indices:
        #         new_centroids = [self.clusters[k].centroid, self.clusters[k].centre_of_mass]
        #         self.clusters = np.delete(self.clusters, k)
        #         self.clusters = np.append(self.clusters, [self.cluster_class(centroid=new_centroids[0],
        #                                                                      other_clusters=self.clusters),
        #                                                   self.cluster_class(centroid=new_centroids[1],
        #                                                                      other_clusters=self.clusters)])
        #
        #         init = True
        #
        #     # check if distance between datapoint and closest cluster is significantly greater than distance between
        #     # other clusters, if so reinitialise todo greater weight for distance to heavy cluster?
        #     # N * K euclidean distance between each data point in data_buf and each existing centroid
        #     data_point_centroid_distances = euclidean_distances(self.data_buf, [c.centroid for c in self.clusters])
        #     # N * 1 distance to closest centroid for each data point in data_buf
        #     min_data_point_centroid_distance = np.min(data_point_centroid_distances, axis=1)
        #     # K * K lower triangular matrix of euclidean distance between each pair of centroids
        #     inter_centroid_distances = np.tril(pairwise_distances([c.centroid for c in self.clusters]))
        #     inter_centroid_distances = inter_centroid_distances[np.where(inter_centroid_distances != 0)]
        #
        #     for d in range(len(self.data_buf)):
        #         distances = np.concatenate([[min_data_point_centroid_distance[d]], inter_centroid_distances])
        #         z = np.abs(stats.zscore(distances, axis=None))
        #         if z[0] > self.outlier_threshold:
        #             self.clusters = np.append(self.clusters,
        #                                       [self.cluster_class(centroid=self.data_buf[i],
        #                                                           other_clusters=self.clusters)])
        #             init = True
        #
        # # check if any clusters have a significantly low mass, if so reinitialise
        # cluster_masses = [c.mass for c in self.clusters]
        # z = np.abs(stats.zscore(cluster_masses, axis=None))
        #
        # outlier_indices = np.concatenate(np.asarray(z > self.outlier_threshold).nonzero())
        # for k in outlier_indices:
        #     self.clusters = np.delete(self.clusters, k)
        #     init = True
        #
        #     # TODO replace these refs with up to date cluster attributes
        #     self.num_clusters = len(self.clusters)
        #     self.centroids = np.vstack([c.centroid for c in self.clusters])
        #
        #     if init:
        #         self.initialise_clusters(num_clusters)
        #         # flush old data from memory
        #         self.flush_data()
        #
        #     elif (len(self.data_buf) + len(self.centroids) >= self.num_clusters) \
        #             and (len(self.data_buf) == self.batch_size):
        #
        #         self.fit_clusters(num_clusters)
        #
        #         # flush old data from memory
        #         self.flush_data()
        #
        # return self.clusters


# class PeriodicOptimalKMeansPlus(ClusteringAlgorithm):
#
#     def __init__(self, clustering_results, features, clusters, init_num_clusters, batch_size, init_batch_size):
#         super().__init__(clustering_results, features, clusters, init_num_clusters, batch_size, init_batch_size)
#
#         self.init_clustering_class = KMeans
#         self.init_clustering_class_args = {'init': 'k-means++'}
#         self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
#
#         self.clustering_class = KMeans
#         self.clustering_class_args = {'n_init': 1}
#         self.clustering_obj = self.clustering_class(**self.clustering_class_args)
#
#         self.cluster_class = GravitationalCluster
#
#     def calc_optimal_num_clusters(self):
#         # cluster parameters: mean vector, covariance matrix
#         # self.clusters[k].centroid
#         # self.clusters[k].covariance
#
#         alpha = 0.05
#         iter_count = 0
#         max_iter = 100000
#
#         # hypothesize initial number of clusters
#         opt_num_clusters = 2
#
#         while iter_count < max_iter:
#
#             # classify dataset into this many clusters
#             self.initialise_clusters()
#
#             # bootstrapped dataset sse values for num_clusters==k
#             syn_dataset_sse_low = np.array([])
#             # bootstrapped dataset sse values for num_clusters==k+1
#             syn_dataset_sse_high = np.array([])
#             num_syn_datasets = 100
#             for s in range(2 * num_syn_datasets):
#
#                 if s < num_syn_datasets:
#                     opt_num_clusters = self.num_clusters
#                 else:
#                     opt_num_clusters = self.num_clusters + 1
#
#                 # generate synthetic dataset
#                 syn_dataset, syn_dataset_labels = make_blobs(
#                     n_samples=K ** 2, n_features=len(self.features), centers=self.centroids,
#                     cluster_std=[np.sqrt(np.diag(c.covariance)) for c in self.clusters])
#
#                 # calculate sse
#                 sse = np.sum([np.sum([np.sum((syn_dataset[i] - self.clusters[k].centroid) ** 2)
#                                       for i in range(len(syn_dataset)) if syn_dataset_labels[i] == k])
#                               for k in range(K)])
#
#                 if s < num_syn_datasets:
#                     syn_dataset_sse_low = np.append(syn_dataset_sse_low, sse)
#                 else:
#                     syn_dataset_sse_high = np.append(syn_dataset_sse_high, sse)
#
#             # calculate probability of each sse occurring in synthetic datasets for low (k number of clusters)
#             # and high (k+1 number of clusters)
#             syn_dataset_sse_low_pdf = stats.norm.pdf(syn_dataset_sse_low)
#             p_value = stats.norm.cdf(syn_dataset_sse_low_pdf)[np.mean(syn_dataset_sse_high)]
#
#             # if the pvalue is greater than the given certainty level, optimal number of clusters has been found
#             if p_value >= alpha:
#                 break
#             # otherwise the optimal number has not yet been found, increment the number of clusters and re-iterate
#             else:
#                 # increment ideal number of clusters
#                 opt_num_clusters += 1
#                 iter_count += 1
#
#     def update_clusters(self, i):
#
#         # if any of the given data is nan, flush it
#         if np.isnan(self.data_buf).any():
#             self.flush_data()
#         # check if clusters have not yet been initialised for the first time
#         elif len(self.clusters) == 0:
#             if len(self.data_buf) == self.num_clusters + 1:
#                 # initialise clusters for the k-1, k and k+1 cases
#                 self.initialise_clusters(self.num_clusters - 1)
#                 self.initialise_clusters(self.num_clusters)
#                 self.initialise_clusters(self.num_clusters + 1)
#                 # flush old data from memory
#                 self.flush_data()
#         else:
#             # check if centroids should be reinitialised based on declining cluster quality
#             init = False
#
#             # sum sse of all clusters at each step in window todo should correspond to the same time steps
#             sse = np.sum([c.sse_window for c in self.clusters], axis=0)
#             # if cluster quality has declined significantly since lass re-initialisation i.e
#             # if sse has continued to climb
#             if (np.diff(sse) > 0).all():
#                 # must reinitialise clusters
#                 init = True
#
#             # classify dataset into K-1, K, K+1 clusters
#             num_clusters = self.calculate_optimal_num_clusters()
#
#             # if existing number of clusters is greater than the optimal number
#             # calculate the resulting sse of each possible merged pair, merge the pair resulting in lowest
#             # sse, and repeat the process iteratively until Kexisting = Kopt
#
#             # elseif existing number of clusters is greater than the optimal number
#             # calculate the resulting sse of each possible split cluster based on upper and lower quartile,
#             # split the cluster resulting in lowest
#             # sse, and repeat the process iteratively until Kexisting = Kopt
#
#             # re - initialise
#             if init:
#                 self.initialise_clusters()
#                 # flush old data from memory
#                 self.flush_data()
#
#             elif (len(self.data_buf) + len(self.centroids) >= self.num_clusters) \
#                     and (len(self.data_buf) == self.batch_size):
#
#                 self.converge()
#
#                 # flush old data from memory
#                 self.flush_data()
#
#         return self.clusters
#
#     def fit(self, init=False):
#
#         centre_of_masses = [c.centre_of_mass for c in self.clusters]
#         cluster_masses = [c.mass for c in self.clusters]
#         new_data_weight = 1 if len(cluster_masses) == 0 else np.min(cluster_masses)
#         weights = np.hstack([np.ones(len(self.data_buf)) * new_data_weight, cluster_masses, cluster_masses])
#         data_points = np.vstack([self.data_buf, self.centroids, centre_of_masses]) if len(self.centroids) \
#             else self.data_buf
#
#         if init:
#             return self.init_clustering_obj.fit(data_points, sample_weight=weights)
#         else:
#             return self.clustering_obj.fit(data_points, sample_weight=weights)
#
#     def update_clustering_obj(self, num_clusters, centroids):
#         self.clustering_class_args.update([('n_clusters', num_clusters), ('init', centroids)])
#         self.clustering_obj = self.clustering_class(**self.clustering_class_args)
#
#     def update_init_clustering_obj(self, num_clusters):
#         self.init_clustering_class_args.update([('n_clusters', num_clusters)])
#         self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
#
#     def initialise_clusters(self, num_clusters):
#         """
#         given a number of clusters, a buffer of data and the real known range of any bounder features,
#         initialise the clusters for this data buffer with Kmeans++
#         """
#
#         # fit = KMeans(n_clusters=self.num_clusters, init='k-means++').fit(self.data_buf)
#         self.update_init_clustering_obj(num_clusters)
#         init_clustering_fit = self.fit(init=True)
#         centroids = init_clustering_fit.cluster_centers_
#
#         f_known = [f for f in range(len(self.features)) if
#                    self.features[f].live and self.features[f].lb is not None]
#
#         # TODO carry old data
#         clusters = np.array([])
#         for k in range(num_clusters):
#             for f in f_known:
#                 centroids[k, f] = np.floor((self.features[f].lb +
#                                             ((self.features[f].ub - self.features[f].lb) * random()))
#                                            / self.features[f].step) * self.features[f].step
#
#             clusters = np.append(clusters, self.cluster_class(centroid=centroids[k]))
#
#         # update centr-of-means of new clusters
#         for k in range(num_clusters):
#             clusters[k].update_com(other_clusters=[clusters[kk] for kk in range(num_clusters) if k != kk])
