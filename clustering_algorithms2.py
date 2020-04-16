import numpy as np
from numpy.random import random, multivariate_normal
from numpy.linalg import norm
from datetime import datetime
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


class ClusterSet:

    def __init__(self, clusters):

        self.clusters = clusters
        self.num_clusters = len(clusters)
        self.centroids = np.array([cluster.centroid for cluster in clusters])
        self.centre_of_masses = np.array([cluster.centre_of_mass for cluster in clusters])
        self.counts = np.array([cluster.count for cluster in clusters])
        self.masses = np.array([cluster.mass for cluster in clusters])
        self.sses = np.array([cluster.sse for cluster in clusters])
        self.covariances = np.array([cluster.covariance for cluster in clusters])


    def update(self):
        self.centroids = np.array([cluster.centroid for cluster in self.clusters])
        self.centre_of_masses = np.array([cluster.centre_of_mass for cluster in self.clusters])
        self.counts = np.array([cluster.count for cluster in self.clusters])
        self.masses = np.array([cluster.mass for cluster in self.clusters])
        self.sses = np.array([cluster.sse for cluster in self.clusters])
        self.covariances = np.array([cluster.covariance for cluster in self.clusters])

    def kill_clusters(self, indices):

        self.clusters = np.delete(self.clusters, indices)
        self.centroids = np.delete(self.centroids, indices)
        self.centre_of_masses = np.delete(self.centre_of_masses, indices)
        self.counts = np.delete(self.counts, indices)
        self.masses = np.delete(self.masses, indices)
        self.covariances = np.delete(self.covariances, indices)
        self.num_clusters -= len(indices)

    def create_clusters(self, new_clusters):
        for cluster in new_clusters:
            self.clusters = np.append(self.clusters, cluster)
            self.centroids = np.append(self.centroids, cluster.centroid)
            self.centre_of_masses = np.append(self.centre_of_masses, cluster.centre_of_mass)
            self.counts = np.append(self.counts, cluster.count)
            self.masses = np.append(self.masses, cluster.mass)
            self.covariances = np.append(self.covariances, cluster.covariance)
            self.num_clusters += 1


class Cluster:
    def __init__(self, centroid):
        # mean of all data points, updated with each new data point added
        self.centroid = centroid
        self.centre_of_mass = np.empty_like(centroid)
        self.centre_of_mass[:] = np.nan
        self.num_features = len(centroid)
        self.covariance = np.zeros((self.num_features, self.num_features))
        self.sse = 0
        self.count = 1
        self.mass = np.nan
        self.time_last_updated = datetime.now()

    def update_data(self, new_data_point):

        # if a new data point has been added to the cluster, update the count, sse, covariance,
        # and time since last update
        # update current SSE and series of SSE values
        self.sse += np.sum((new_data_point - self.centroid) ** 2)

        # update data point count
        self.count += 1

        # update cluster covariance matrix
        if self.count > 1:
            diff = new_data_point - self.centroid
            self.covariance += (diff * diff.T - self.covariance) / (self.count - 1)

        # update last updated time
        self.time_last_updated = datetime.now()

    def update_centroid(self, new_centroid):
        # if the centroid is being updated, update the centroid coordinates and reset the sse
        # update centroid coordinates
        self.centroid = new_centroid
        self.sse = 0

    def update_centre_of_mass(self, data, member_probabilities):
        pass

    def decay(self):
        pass


class GravitationalCluster(Cluster):

    def __init__(self, centroid, gravitational_const=0.01):
        # mass is sum of data points * time since last addition
        self.mass = 0
        self.gravitational_const = gravitational_const
        # mean of all data points, updated with each new data point added
        self.centroid = centroid
        # centre-of-mass of the cluster, updated with each new data point added
        # mean of data points significantly close to centre-of-mass
        self.centre_of_mass = centroid.copy()
        self.num_features = len(centroid)

        self.covariance = np.zeros((self.num_features, self.num_features))
        self.sse = 0
        self.count = 0
        self.time_last_updated = datetime.now()

    def update_data(self, new_data_point=None, max_count=1):

        # if a new data point was added, update the count, sse and mass
        # update current SSE and series of SSE values
        self.sse += np.sum((new_data_point - self.centroid) ** 2)

        # update data point count
        self.count += 1
        # TODO rather than normalising, set count to 0 on re-initialisation?
        self.mass = self.count / max_count

        # update cluster covariance matrix
        if self.count > 1:
            diff = new_data_point - self.centroid
            self.covariance += (diff * diff.T - self.covariance) / (self.count - 1)

        # update last updated time
        self.time_last_updated = datetime.now()

    def decay(self):
        # TODO should be user inputs
        max_count = 1
        time_decay_const = 100

        # if no data point was added to this cluster during this data feed in, then decay its mass
        if self.count:
            # TODO second time is arbitrary, depends on OS, better to use number of data points received
            self.mass = (self.count / max_count) \
                        * np.exp(-(datetime.now() - self.time_last_updated).total_seconds() / time_decay_const)

    def update_centre_of_mass(self, new_data_points, member_probabilities):
        # if the clusters are gravitational, adjust each cluster's centre of mass by the new data points
        # pull each cluster's centre of mass towards the new data points, depending on weight and distance
        # pull is proportional to distance squared to data point, proportional to degree of certainty that the data point
        # belongs to this cluster, inversely proportional to mass of cluster
        # TODO or only pull centre_of_mass of cluster to which this data point was added?
        self.centre_of_mass += self.gravitational_const \
                               * np.sum((new_data_points - self.centre_of_mass) \
                                        * norm((new_data_points - self.centre_of_mass), ord=2, axis=1)
                                        * member_probabilities, axis=0)

    def update_centroid(self, new_centroid):
        # update centroid coordinates, reset sse
        self.centroid = new_centroid
        self.sse = 0


class Feature:
    def __init__(self, name, lb=None, ub=None, step=None):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.step = step
        self.live = True


class ClusteringAlgorithm:

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size,
                 window_size):

        # N (number of data points) * D (number of dimensions) array of data stream coordinates
        self.data_buf = np.array([])
        self.features = features
        self.batch_size = batch_size
        self.init_batch_size = init_batch_size
        self.window_size = window_size

        self.optimal_num_clusters = init_num_clusters
        self.cluster_set = ClusterSet([])

        self.cluster_class = None
        self.init_clustering_class = None
        self.init_clustering_class_args = None
        self.init_clustering_obj = None  # self.init_clustering_class(**self.init_clustering_class_args)
        self.clustering_class = None
        self.clustering_class_args = None
        self.clustering_obj = None  # self.clustering_class(**self.clustering_class_args)

    def initialise_clusters(self, data, cluster_set, optimal_num_clusters, **kwargs):

        # fit the initial clusters by Kmeans++
        running_time = datetime.now()
        init_clustering_fit = self.fit(data, cluster_set, optimal_num_clusters, init=True)
        running_time = (datetime.now() - running_time).total_seconds()
        num_iters = self.init_clustering_obj.n_iter_

        centroids = init_clustering_fit.cluster_centers_

        f_known = [f for f in range(len(self.features)) if
                   self.features[f].live and self.features[f].lb is not None]

        for k in range(cluster_set.num_clusters):
            for f in f_known:
                centroids[k, f] = np.floor((self.features[f].lb +
                                            ((self.features[f].ub - self.features[f].lb) * random()))
                                           / self.features[f].step) * self.features[f].step

        cluster_set = ClusterSet(np.array([self.cluster_class(centroids[k], **kwargs)
                                           for k in range(optimal_num_clusters)]))

        sse = np.sum(cluster_set.sses)

        return cluster_set, running_time, num_iters, sse

    def fit_clusters(self, data, cluster_set, optimal_num_clusters):

        # fit buffered data points to clusters and time it
        running_time = datetime.now()
        fit = self.fit(data, cluster_set, optimal_num_clusters, init=False)
        running_time = (datetime.now() - running_time).total_seconds()
        num_iters = self.clustering_obj.n_iter_

        # update cluster centroids and centre-of-masses
        centroids = fit.cluster_centers_
        closest_cluster_distances = np.argmin(fit.transform(data), axis=1)

        # update cluster data point parameters and centre-of-masses
        # loop through all existing clusters
        for k in range(cluster_set.num_clusters):
            member_probabilities = np.zeros_like(data)
            # if they have been adjusted, update their centroid and add the data points
            if k in closest_cluster_distances:
                cluster_set.clusters[k].update_centroid(new_centroid=centroids[k])
                # update the cluster params of whichever cluster this new data point was added to
                for d in [d for d in range(len(closest_cluster_distances)) if closest_cluster_distances[d] == k]:
                    # without uncertainty, this data point definitely belongs to this centroid
                    member_probabilities[d] = 1
                    cluster_set.clusters[k].update_data(new_data_point=data[d])

            # if they have not been adjusted, decay the mass
            else:
                cluster_set.clusters[k].decay()


            cluster_set.clusters[k].update_centre_of_mass(data, member_probabilities)

        cluster_set.update()
        sse = np.sum(cluster_set.sses)

        return cluster_set, running_time, num_iters, sse

    def feed_data(self, new_data):

        # if no data has yet been added to buffer, initialise multi-dimensional array
        if len(self.data_buf) == 0:
            self.data_buf = np.expand_dims(new_data[[feat.name for feat in self.features]].values, axis=0)
        # else append new data points
        else:
            self.data_buf = np.vstack((self.data_buf, new_data[[feat.name for feat in self.features]].values))

    def flush_data(self):

        # flush data buffered from memory
        self.data_buf = np.array([])


class KMeansPlus(ClusteringAlgorithm):

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size, window_size):
        super().__init__(features, init_num_clusters, batch_size, init_batch_size, window_size)

        self.init_clustering_class = KMeans
        self.init_clustering_class_args = {'init': 'k-means++'}
        self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)

        self.clustering_class = KMeans
        self.clustering_class_args = {'n_init': 1}
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)

        self.cluster_class = Cluster

    def fit(self, data, cluster_set, optimal_num_clusters, init=False):

        # generate equivalent weights for the new data points and the existing centroids
        weights = np.ones(len(data) + cluster_set.num_clusters)

        # generate stack of coordinates consisting of new data points and existing centroids
        data_points = np.vstack([data, cluster_set.centroids]) \
            if cluster_set.num_clusters else data

        # an initialisation fit (convergence initialised with existing centroids)
        if init:

            # update the initialisation clustering object with a new number of clusters
            self.init_clustering_class_args.update([('n_clusters', optimal_num_clusters)])
            self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
            return self.init_clustering_obj.fit(data_points, sample_weight=weights)
        else:

            # update the clustering object with a new number of clusters and pass existing centroids as initialisation
            self.clustering_class_args.update([('n_clusters', optimal_num_clusters),
                                               ('init', cluster_set.centroids)])
            self.clustering_obj = self.clustering_class(**self.clustering_class_args)
            return self.clustering_obj.fit(data_points, sample_weight=weights)

    def update_clusters(self, i):

        running_time = np.nan
        num_iters = np.nan
        sse = np.nan

        # if any of the given data is nan, flush it
        if np.isnan(self.data_buf).any():
            self.flush_data()
        # else if enough data points and existing clusters are available to fit data
        elif len(self.data_buf) + self.cluster_set.num_clusters >= self.optimal_num_clusters:
            # if enough data points have been buffered to re-initialise clusters
            if (i % self.init_batch_size) == 0:
                # (re)initialise cluster means
                self.cluster_set, running_time, num_iters, sse = \
                    self.initialise_clusters(self.data_buf, self.cluster_set, self.optimal_num_clusters)

                # flush old data from memory
                self.flush_data()
            # else if cluster centres must not yet be (re)initialised, feed data to stream to update clusters
            elif len(self.data_buf) == self.batch_size:
                self.cluster_set, running_time, num_iters, sse = \
                    self.fit_clusters(self.data_buf, self.cluster_set, self.optimal_num_clusters)

                # flush old data from memory
                self.flush_data()

        return self.cluster_set, running_time, num_iters, sse


class GravitationalKMeansPlus(ClusteringAlgorithm):

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size,
                 outlier_threshold=2, gravitational_const=0.1, window_size=100):
        super().__init__(features, init_num_clusters, batch_size, init_batch_size, window_size)

        self.init_clustering_class = KMeans
        self.init_clustering_class_args = {'init': 'k-means++'}
        self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)

        self.clustering_class = KMeans
        self.clustering_class_args = {'n_init': 1}
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)

        self.cluster_class = GravitationalCluster

        self.outlier_threshold = outlier_threshold
        self.gravitational_const = gravitational_const

        # define cluster parameters for the K-1, K and K+1 cases
        self.optimal_num_clusters = [init_num_clusters - 1, init_num_clusters, init_num_clusters + 1]
        self.cluster_set = [ClusterSet([]), ClusterSet([]), ClusterSet([])]

    def calc_opt_num_clusters(self):
        """
        given the cluster parameters of datasets clustered into K-1, K and K+1 clusters,
        calculate the optimal number of clusters by generating bootstrap synthetic data sets, calculating
        the pdf of their SSE values and calculating the cd of these pdf functions up to the SSE achieved for the
        next greatest number of clusters
        """

        alpha = 0.05
        new_optimal_num_clusters = None

        num_syn_datasets = np.int(np.sqrt(self.init_batch_size))
        n_samples = self.init_batch_size

        for c in range(2):

            actual_sse = np.sum(self.cluster_set[c + 1].sses)

            # bootstrapped dataset sse values for num_clusters[c]
            synthetic_sses = np.array([])

            for s in range(num_syn_datasets):
                # generate synthetic datasets based on cluster parameters (mean vector and covariance matrix)
                syn_dataset, syn_dataset_labels = make_blobs(
                    n_samples=n_samples, n_features=len(self.features),
                    centers=self.cluster_set[c].centroids,
                    cluster_std=[np.sqrt(np.diag(cov)) for cov in self.cluster_set[c].covariances])

                # cluster the synthetic datasets
                init_clustering_fit = self.fit(syn_dataset, self.cluster_set[c], init=True)
                syn_centroids = init_clustering_fit.cluster_centers_

                # plt.scatter(syn_dataset[:, 0], syn_dataset[:, 1], marker='o', c=syn_dataset_labels,
                #             s=25, edgecolor='k')
                #
                # plt.show()

                # calculate sse
                synthetic_sse = np.sum([np.sum([np.sum((syn_dataset[i] - syn_centroids[k]) ** 2)
                                                for i in range(len(syn_dataset)) if syn_dataset_labels[i] == k])
                                        for k in range(self.optimal_num_clusters[c])])

                synthetic_sses = np.append(synthetic_sses, synthetic_sse)

            # calculate pdf of SSE values achieved over all of these data sets
            # calculate probability of each sse occurring in synthetic datasets for low (k number of clusters)
            # and high (k+1 number of clusters)
            # synthetic_sse_pdf = stats.norm.pdf(synthetic_sses)

            # calculate cdf of SSE pdf at the SSE achieved for the next greatest number of clusters K+1
            p_value = np.interp(actual_sse, synthetic_sses, stats.norm.cdf(synthetic_sses))

            new_optimal_num_clusters = self.optimal_num_clusters[c]
            if p_value > alpha:
                break

        return new_optimal_num_clusters

    def expand_cluster(self):
        """
        find the cluster with the greatest centre-of-mass/centroid euclidean difference and split it
        """
        opt_index = 1
        centroid_com_distances = np.array(
            np.sum((self.cluster_set[opt_index].centroids - self.cluster_set[opt_index].centre_of_mass) ** 2) ** 0.5)
        split_idx = np.argmax(centroid_com_distances)

        new_centroids = [self.cluster_set[opt_index][split_idx].centroid,
                         self.cluster_set[opt_index][split_idx].centre_of_mass]
        self.cluster_set[opt_index].kill_cluster(split_idx)
        self.cluster_set[opt_index].create_cluster([self.cluster_class(centroid=new_centroids[0],
                                                                       gravitational_const=self.gravitational_const),
                                                    self.cluster_class(centroid=new_centroids[1],
                                                                       gravitational_const=self.gravitational_const)])

    def reduce_cluster(self):
        """
        find the cluster with the lowest mass
        or the pair of clusters with the lowest centre-of-mass euclidean difference and merge them
        """

        opt_index = 1

        # check if any clusters have a significantly low mass, kill it if so
        mass_z = np.abs(stats.zscore(self.cluster_set[opt_index].masses, axis=None))

        # outlier_indices = np.concatenate(np.asarray(z > self.outlier_threshold).nonzero())
        mass_outlier_index = np.argmin(mass_z)

        # check if any centre of masses are significantly close and merge them if so
        com_distances = np.tril(pairwise_distances(self.cluster_set[opt_index].centre_of_masses))
        com_distances = com_distances[np.where(com_distances != 0)]
        com_z = np.abs(stats.zscore(com_distances, axis=None))
        com_outlier_index = np.argmax(com_z)

        # if the z-value associated with the lightest cluster is greatest, kill the lightweight cluster
        if mass_z[mass_outlier_index] > com_z[com_outlier_index]:
            #self.clusters[c] = np.delete(self.cluster_set[opt_index], mass_outlier_index)
            self.cluster_set[opt_index].kill_cluster(mass_outlier_index)
        # otherwise merge the close clusters
        else:
            new_centroid = np.mean(self.cluster_set[opt_index].centroids[com_outlier_index], axis=0)

            self.cluster_set[opt_index].kill_cluster(com_outlier_index)
            self.cluster_set[opt_index].create_cluster(self.cluster_class(centroid=new_centroid,
                                                                          gravitational_const=self.gravitational_const))

            # self.clusters[c] = np.delete(self.cluster_set[opt_index], com_outlier_index)
            # self.clusters[c] = np.append(self.cluster_set[opt_index],
            #                              self.cluster_class(centroid=new_centroid,
            #                                                 gravitational_const=self.gravitational_const,
            #                                                 window_size=self.window_size))

    def update_clusters(self, i):

        running_time = np.nan(3)
        num_iters = np.nan(3)
        sse = np.nan(3)
        opt_index = 1

        # if any of the given data is nan, flush it
        if np.isnan(self.data_buf).any():
            self.flush_data()
        # check if clusters have not yet been initialised
        elif self.cluster_set[opt_index].num_clusters == 0:
            # if enough data points have been buffered to initialise clusters for the first time
            if len(self.data_buf) == self.optimal_num_clusters[-1]:
                for c in range(3):
                    # the clusters are re-initialised (counts, sse, centroids and masses are reset)
                    self.cluster_set[c], running_time[c], num_iters[c], sse[c] = \
                        self.initialise_clusters(self.data_buf, self.cluster_set[c], self.optimal_num_clusters[c],
                                                 gravitational_const=self.gravitational_const,
                                                 window_size=self.window_size)

                # flush old data from memory
                self.flush_data()
        else:
            # TODO the pull of a new data point should be proportional to the square of its distance to closest com
            #  such that an outlier can have a significant pull on the centre of masses (but only on closest centroid)

            # TODO the mass of a cluster should equal sum(r^-2) fpr all data points in cluster such that the mass is
            #   greater for tightly-packed centroids

            # TODO the pull of a new data point on a given centre-of-mass should be proportional to the liklihood that
            #   it belongs to that cluster


            # else check if centroids should be reinitialised based on declining cluster quality
            # sum sse of all clusters at each step in window todo should correspond to the same time steps
            # sse_diff = np.diff(np.sum([cluster.sse_window for cluster in self.clusters[1]], axis=0), n=2)

            # if the minimum distance from data point to centroid is significantly large,
            # or the minimum distance between two centroids is significantly close
            # check distances between new points and existing centroids
            # point_centroid_distances = euclidean_distances(self.data_buf, self.cluster_set[opt_index].centroids)
            # min_point_centroid_distance = np.min(point_centroid_distances, axis=1)
            # # if this is significantly higher than the highest inter-centroidal distance,
            # # check for higher number of clusters
            #
            # # check distances between existing centroids
            # inter_centroid_distances = np.tril(euclidean_distances(self.cluster_set[opt_index].centroids))
            # inter_centroid_distances = inter_centroid_distances[np.where(inter_centroid_distances != 0)]
            # max_inter_centroid_distance = np.max(inter_centroid_distances)

            # if cluster quality has declined significantly since last re-initialisation
            # ie if the minimum distance from any of these data points to their closest centroid is greater than
            # the maximum distance between centroids for several iterations
            # todo how to tell if cluster number should be re-evaluated? better to do periodically?
            # if (min_point_centroid_distance > max_inter_centroid_distance).any():
            if (i % self.init_batch_size) == 0:
                # must reinitialise clusters
                new_optimal_num_clusters = self.calc_opt_num_clusters()

                # if the optimal number of clusters is greater than the current number of clusters,
                # split the cluster with the highest centroid, centre-of-mass difference and re-initialise based
                # on these centroids by KMeans++
                if new_optimal_num_clusters > self.optimal_num_clusters[opt_index]:
                    self.expand_cluster()
                elif new_optimal_num_clusters < self.optimal_num_clusters[opt_index] and new_optimal_num_clusters > 1:
                    self.reduce_cluster()

                self.optimal_num_clusters = [new_optimal_num_clusters - 1,
                                             new_optimal_num_clusters,
                                             new_optimal_num_clusters + 1]

                for c in range(3):
                    self.cluster_set[c], running_time[c], num_iters[c], sse[c] = \
                        self.initialise_clusters(self.data_buf, self.cluster_set[c], self.optimal_num_clusters[c],
                                                 gravitational_const=self.gravitational_const,
                                                 window_size=self.window_size)

                # flush old data from memory
                self.flush_data()

            # else if cluster centres must not yet be (re)initialised, feed data to stream to update clusters
            elif len(self.data_buf) == self.batch_size:
                for c in range(3):
                    self.cluster_set[c], running_time[c], num_iters[c], sse[c]  = \
                        self.fit_clusters(self.data_buf, self.optimal_num_clusters[c], self.cluster_set[c])

                # flush old data from memory
                self.flush_data()

        return self.cluster_set[opt_index], running_time[opt_index], num_iters[opt_index], sse[opt_index]


    def fit(self, data, cluster_set, optimal_num_clusters, init=False):

        # if clusters already exist, each new data point is given a weight of 1 and each existing cluster is given a
        # weight equal to its mass

        if cluster_set.num_clusters:
            data_points = np.concatenate([data, cluster_set.centroids])
            weights = np.concatenate([np.ones(len(data)), cluster_set.masses])

        else:
            data_points = data
            weights = np.ones(len(data))

        if init:

            # update the initialisation clustering object with a new number of clusters
            self.init_clustering_class_args.update([('n_clusters', optimal_num_clusters)])
            self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
            return self.init_clustering_obj.fit(data_points, sample_weight=weights)
        else:

            # update the clustering object with a new number of clusters and pass existing centroids as initialisation
            self.clustering_class_args.update([('n_clusters', optimal_num_clusters),
                                               ('init', cluster_set.centroids)])
            self.clustering_obj = self.clustering_class(**self.clustering_class_args)
            return self.clustering_obj.fit(data_points, sample_weight=weights)
