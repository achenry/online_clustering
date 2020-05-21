import numpy as np
from numpy.linalg import norm
from datetime import datetime
from FuzzyKMeansPlus import FuzzyKMeansPlus
from sklearn.cluster import OPTICS
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from copy import deepcopy

"""
NOTES
Random initialisation within know limits in parallel to check for better results
or Planting seeds.at known limits of features to influence centre of masses

To buffer data for re-init or not

COMs are pulled toward new data points by a gravity that is proportional to membership probability/(mass * r^2)

Evolutionary metrics such as mass, sse and count are probabilistic, carried over by new initialisations and account for 
historic data, we assume that additions and subtractions cancel over time,
  how best to measure and apply online estimates

Evolving parameters: n_clusters, gravitational constant, alpha

Evolving Periodic KMeans: Could randomly increase/decrease and check for improvement, then do opposite for re-initialisation

Vary weight given to outliers

Flaws with original KMeans:
Centroids can tend towards eachother until new initialisation. 
Centroids which are set to outliers from initialisation will remain so for some time.

Parameterising the time_decay variable in order to select time windows of interest and then to kill outlying clusters

Performance drops for high number of clusters.

Implement hierarchical merging/splitting techniques to merge/split. ie Ward criterion.

Visible (initialise clusters run again to view and merge/kill/split any clusters as necessary) vs invisible clusters

Single pass

Kill lightweight cluster after time-decay without checking opt K as with create

"""

"""
CASE STUDIES:
1) Offline PeriodicKMeansPlus and OptimalKMeansPlus to tune parameters gravitational_const, alpha
2) Vary batch_size from 1 to 48
3) Vary g from 0 to 1
4) Vary fuzziness from 1 to 4
5) Vary create_clusters boolean to check if it is necessary to explicitly create cluster from outlier or if it will 
    happen as a result of diverging coms
6) Vary kill_clusters boolean to check if it is necessary to explicitly kill lightweight cluster or if it will be pulled 
   towards heavier clusters and merged anyway.

for three customer data sets each
"""

"""
CODE:
Comment
Allow for variable data inputs from params (ie number of data points)
"""


class ClusterSet:
    """
    Class defining a set of clusters and their attributes for a given problem
    """

    def __init__(self, clusters):
        """
        :param clusters: list of Cluster objects which make up this ClusterSet
        """

        self.num_clusters = len(clusters)

        # lists of most up-to-date attributes clusters. Eahc list is of length equal to the number of clusters
        self.clusters = clusters
        self.centroids = np.array([cluster.centroid for cluster in clusters])
        self.centre_of_masses = np.array([cluster.centre_of_mass for cluster in clusters])
        self.counts = np.array([cluster.count for cluster in clusters])
        self.fuzzy_counts = np.array([cluster.fuzzy_count for cluster in clusters])
        self.masses = np.array([cluster.mass for cluster in clusters])
        # self.sses = np.array([cluster.sse for cluster in clusters])
        # self.covariances = np.array([cluster.covariance for cluster in clusters])
        # self.variances = np.array([cluster.variance for cluster in clusters])
        self.variances_since_last_init = []
        self.counts_since_last_init = []
        self.fuzzy_counts_since_last_init = []

        # list of total cluster set sse (summed over all clusters), compactness (averaged over all clusters) and
        # dbi (averaged over all clusters) for each algorithm run since clusters were last
        # re-initialised in initialise_clusters function
        self.sse_since_last_init = []
        self.cp_since_last_init = []
        self.dbi_since_last_init = []

    def update(self, fuzzy_weights):
        """
        Update the ClusterSet object each time clusters are re-fitted in response to each new data point,
        when fit_clusters is called
        :param fuzzy_labels: 1 * K (number of clusters) array giving the probability that the most recent
                                      incoming data point belongs to each cluster
        """

        # update all cluster attributes
        self.centroids = np.array([cluster.centroid for cluster in self.clusters])
        self.centre_of_masses = np.array([cluster.centre_of_mass for cluster in self.clusters])
        self.counts = np.array([cluster.count for cluster in self.clusters])
        self.fuzzy_counts = np.array([cluster.fuzzy_count for cluster in self.clusters])
        self.masses = np.array([cluster.mass for cluster in self.clusters])
        # self.sses = np.array([cluster.sse for cluster in self.clusters])
        # self.covariances = np.array([cluster.covariance for cluster in self.clusters])
        # self.variances = np.array([cluster.variance for cluster in self.clusters])
        self.variances_since_last_init = np.array([cluster.variance_since_last_init for cluster in self.clusters])
        self.counts_since_last_init = np.array([cluster.count_since_last_init for cluster in self.clusters])
        self.fuzzy_counts_since_last_init = np.array([cluster.fuzzy_count_since_last_init for cluster in self.clusters])

        # update all cluster dbi values, if more than one cluster exists
        if self.num_clusters > 1:
            for k, cluster1 in enumerate(self.clusters):
                dbi = np.max([(cluster1.cp_since_last_init + cluster2.cp_since_last_init)
                              / norm(cluster1.centroid - cluster2.centroid, 2)
                              for kk, cluster2 in enumerate(self.clusters) if kk != k])

                cluster1.update_metrics(dbi=dbi, fuzzy_weight=fuzzy_weights[k])

        # add a new total sse, mean compactness or mean dbi value to the aggregate window lists, corresponding to this
        # re-fit
        self.sse_since_last_init.append(np.sum([cluster.sse_since_last_init for cluster in self.clusters]))
        self.cp_since_last_init.append(np.mean([cluster.cp_since_last_init for cluster in self.clusters]))
        self.dbi_since_last_init.append(np.mean([cluster.dbi_since_last_init for cluster in self.clusters]))

        # update the number of clusters
        self.num_clusters = len(self.clusters)

    def kill_clusters(self, indices):
        """
        Remove the clusters at the given indices and all attributes associated with them.
        :param indices: a list of indices, at which clusters should be removed from this cluster set
        """

        self.clusters = np.delete(self.clusters, indices, axis=0)
        self.centroids = np.delete(self.centroids, indices, axis=0)
        self.centre_of_masses = np.delete(self.centre_of_masses, indices, axis=0)
        self.counts = np.delete(self.counts, indices, axis=0)
        self.fuzzy_counts = np.delete(self.fuzzy_counts, indices, axis=0)
        self.masses = np.delete(self.masses, indices, axis=0)
        # self.covariances = np.delete(self.covariances, indices, axis=0)
        # self.variances = np.delete(self.variances, indices, axis=0)
        # self.sses = np.delete(self.sses, indices, axis=0)

        # decrement the number of clusters
        self.num_clusters -= len(indices)

    def create_clusters(self, new_clusters):
        """
        Add each of the list of given clusters and all of their attributes to this cluster set.
        :param new_clusters: list of Cluster objects
        """
        # add attributes of each new cluster the this cluster set's attributes
        for cluster in new_clusters:
            self.clusters = np.append(self.clusters, cluster)
            self.centroids = np.vstack([self.centroids, cluster.centroid])
            self.centre_of_masses = np.vstack([self.centre_of_masses, cluster.centre_of_mass])
            self.counts = np.append(self.counts, cluster.count)
            self.fuzzy_counts = np.append(self.fuzzy_counts, cluster.fuzzy_count)
            self.masses = np.append(self.masses, cluster.mass)
            # self.covariances = np.vstack([self.covariances, [cluster.covariance]])
            # self.variances = np.vstack([self.variances, [cluster.variance]])
            # self.sses = np.append(self.sses, cluster.sse)

        # increment number of clusters
        self.num_clusters += len(new_clusters)


class Cluster:
    """
    Parent Class defining a cluster, which represents a group of similar data points.
    """

    def __init__(self, centroid, sample_count_initialised):
        """
        :param centroid: given centroid at which this cluster center is located
        """

        # centroid is the mean of all data points, updated with each new data point added
        self.centroid = centroid

        # centre-of-mass is originally located at the centroid, but if gravitational_const > 0, then it will be
        # pulled towards new data points by the cluster method update_centre_of_mass
        self.centre_of_mass = np.nan

        # number of features, or dimension the data points represented by this cluster
        self.num_features = len(centroid)

        # covariance between different features of the data points represented by this cluster
        # self.covariance = np.zeros((self.num_features, self.num_features))
        self.variance_since_last_init = np.zeros(self.num_features)
        self.count_since_last_init = 0

        # sum sum-squared error, mean compactness, mean dbi of cluster since last initialisation
        self.sse_since_last_init = 0
        self.cp_since_last_init = 0
        self.dbi_since_last_init = 0

        # up-to-date count of cluster = number of data points added to this cluster.
        # This value is normalised upon each new cluster initialisation by subtracting the lowest count of all clusters
        # In online algorithms, this value is inaccurate, as data points which originally belongs to this cluster
        # may not anymore, and vice-versa, since the centroid has moved.
        self.count = 0

        # up-to-date probabilistic count of cluster = sum of membership probability (liklihood that the data point
        # belongs to this cluster) of each incoming data point.
        # This value is normalised upon each new cluster initialisation by subtracting the lowest probabilistic count
        # of all clusters.
        # In online algorithms, this value is inaccurate, as data points which were likely to belong to this cluster
        # may now be less likely, and vice-versa, since the centroid has moved.
        self.fuzzy_count = 0
        self.fuzzy_count_since_last_init = 0

        # the mass of a cluster is based on its probabilistic count.
        # Its value decays exponentially with each re-fit in which no incoming data point is closest to it
        # (ie highest membership probability)
        self.mass = 0
        self.sample_count_last_updated = sample_count_initialised

    def update_metrics(self, sse=None, dbi=None, compactness=None, fuzzy_label=None, fuzzy_weight=None):
        """
        Update the metrics of this cluster with each incoming data point.
        Note that the given sse, dbi, compactness values are based only on the most recent incoming data points.
        :param sse: new Sum-Squared error (or inertia) from this re-fit
        :param dbi: new dbi  from this re-fit
        :param compactness:
        :param fuzzy_label:
        :return:
        """

        if sse is not None and fuzzy_weight:
            self.sse_since_last_init += sse * fuzzy_weight

        if dbi is not None:
            self.dbi_since_last_init = dbi

        if compactness is not None and fuzzy_weight and self.fuzzy_count_since_last_init:
            self.cp_since_last_init += ((fuzzy_weight * compactness) - (fuzzy_label * self.cp_since_last_init)) / \
                                       self.fuzzy_count_since_last_init

    def update_data(self, sample_count, new_data_point, fuzzy_label, fuzzy_weight, is_closest, tol):
        """
        Update the count, probabilistic count, covariance, iteration since last update 
        of this cluster with each incoming data point.
        :param sample_count: integer i for the i'th batch if data points delivered to the algorithm since the start
        :param new_data_point: 1 * F (number of features) ndarray
        :param fuzzy_label: scalar liklihood that new_data_point belongs to this cluster
        :param fuzzy_weight: fuzzy_label raised to the value of fuzziness factor
        :param is_closest: boolean indicating if the new data point is closer to this cluster than the others 
                           (ie highest membership probability)
        """

        # if a new data point has been added to the cluster, update the count, sse, covariance,
        # and time since last update

        # if this data point was closest to this cluster's center

        if is_closest:
            # increment data point count
            self.count += 1
            self.count_since_last_init += 1
            # update data count at which this cluster has most recently been updated (only on hard assignment)
            self.sample_count_last_updated = sample_count

        # increment fuzzy_count with the probability that this data point belongs to this cluster
        self.fuzzy_count += fuzzy_label
        self.fuzzy_count_since_last_init += fuzzy_label

        # update covariance matrix
        if self.fuzzy_count_since_last_init > 1:
            diff = new_data_point - self.centroid
            # self.covariance += ((diff * diff.T) - self.covariance) / (self.count - 1)

            # this variance is only used to synthesise datasets, from which inertia is calculated in a fuzzy manner

            self.variance_since_last_init += ((fuzzy_weight * diff ** 2) -
                                              (fuzzy_label * self.variance_since_last_init)) / \
                                             (self.fuzzy_count_since_last_init - 1)
            #
            # self.variance_since_last_init += ((diff ** 2) -
            #                                   (1 * self.variance_since_last_init)) / \
            #                                  (self.count_since_last_init - 1)

    def update_centroid(self, new_centroid):
        """
        Update the coordinates of the centroid with each new incoming data point.
        :param new_centroid: 1 * F (number of features) ndarray
        """
        # update centroid coordinates
        self.centroid = new_centroid

    def update_centre_of_mass(self, data, fuzzy_label, tol):
        pass

    def decay(self, time_decay, sample_count, is_closest):
        pass


class GravitationalCluster(Cluster):
    """
    Clustering object with a) centre-of-mass attribute which is influenced by the gravitational pull of new data points,
    generally follow trends in the data that are not captured by the centroid and are used to decide which clusters
    should be merged or split and b) mass attribute which decays with time as the cluster is not updated and is used
    to kill significantly lightweight clusters and to move lightweight clusters towards new data trends in order to 
    merge them or split them if necessary.
    """

    def __init__(self, centroid, sample_count_initialised=0, gravitational_const=1, time_decay_const=48):
        super().__init__(centroid=centroid, sample_count_initialised=sample_count_initialised)
        # proportionality constant used when pulling centre-of-masses towards new data points
        self.gravitational_const = gravitational_const
        self.time_decay_const = time_decay_const
        # centre-of-mass of the cluster, updated with each new data point added
        self.centre_of_mass = centroid.copy()

    def update_data(self, sample_count, new_data_point, fuzzy_label, fuzzy_weight, is_closest, tol):
        """
        see definition in Cluster Class
        """
        super().update_data(sample_count=sample_count, new_data_point=new_data_point,
                            fuzzy_label=fuzzy_label, fuzzy_weight=fuzzy_weight, is_closest=is_closest,
                            tol=tol)

        if is_closest:
            self.mass += fuzzy_label
        else:
            # if no incoming data point closer to this centroid than all others, then the mass is exponentially decayed
            # a mass of 1 will be reduced to self.tol within self.time_decay_const number of samples if it is not
            #added to by a closest data point
            self.mass *= tol ** (1 / self.time_decay_const)
                #np.max([1, self.fuzzy_count * 0.95])
                                #np.exp(-(sample_count - self.sample_count_last_updated) / time_decay)])

    def update_centre_of_mass(self, new_data_points, fuzzy_labels, tol):
        """
        Update the centre of mass coordinates with each new incoming batch of data points based on the gravitational
        pull of these new data points on the centre of mass, depending on mass and distance. 
        The location of the centre of mass will give us an indication as to which clusters should be split and merged 
        later.
        :param new_data_points: N (batch size of data points) * F (number of features) ndarray
        :param fuzzy_labels: N * 1 ndarray giving liklihood that each data point belongs to this cluster
        :param tol: minimum tolerated nonzero change in centre of mass, gravitational constant must be changed if this
                    is violated
        :return: 
        """
        if self.gravitational_const == 0:
            self.centre_of_mass = self.centroid
        else:
            euc_dist = norm((new_data_points - self.centre_of_mass), ord=2, axis=1)

            # loop through the batch of data points
            for d in range(len(new_data_points)):

                # if there is a nonzero distance between the new data point and the centre of mass
                if euc_dist[d]:

                    # gravitational pull is towards new data point
                    com_dir_change = (new_data_points[d] - self.centre_of_mass) / euc_dist[d]

                    # gravitational pull is inversely proportional to distance squared to data point,
                    # proportional to degree of certainty that the data point belongs to this cluster
                    # and inversely proportional to the mass of the cluster
                    com_mag_change = np.min([euc_dist[d], self.gravitational_const * fuzzy_labels[d] /
                                             (self.mass * euc_dist[d] ** 2)])

                    if com_mag_change > 0 and (com_mag_change < tol < euc_dist[d]):
                        com_mag_change = tol

                        # gravitational constant should tend to 0 as the com approaches the data point
                        # update_com_mag_change = False
                        # if com_mag_change > euc_dist[d]:
                        #
                        #     # if the magnitude change would pull the com beyond the new data point,
                        #     # reduce the gravitational_constant exponentially by at most as much as would be required
                        #     # to just about bring the com to the data point
                        #     temp_gravitational_const = np.min([self.gravitational_const * np.exp(-(1 / euc_dist[d])),
                        #                                        (self.mass * euc_dist[d] ** 2) / fuzzy_labels[d]])
                        #     update_com_mag_change = True
                        # if com_mag_change < tol:
                        #     # if the magnitude change would barely pull the com from its current position,
                        #     # increase the gravitational_constant exponentially by at least as much as would be required
                        #     # to pull the com by tolerance amount
                        #     temp_gravitational_const = np.max([self.gravitational_const * np.exp(1 / euc_dist[d]),
                        #                                        (tol * euc_dist[d] * self.mass * euc_dist[d] ** 2)
                        #                                        / fuzzy_labels[d]])
                        # update_com_mag_change = True

                    # update the com change if necessary to reflect changes in gravitational_constant
                    # if update_com_mag_change:
                    #     com_mag_change = (temp_gravitational_const *
                    #                      fuzzy_labels[d] / (self.mass * euc_dist[d] ** 2)) / euc_dist[d]

                    com_change = com_dir_change * com_mag_change
                    self.centre_of_mass += com_change

class Feature:
    """
    Class defining Feature object associated with incoming data source
    """

    def __init__(self, name, lb=None, ub=None, step=None):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.step = step
        self.live = True


class ClusteringAlgorithm:
    """
    Parent Class defining clustering algorithm.
    """

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size=None, fuzziness=2,
                 max_iter=100000, tol=0.000001, alpha=None, gravitational_const=None, time_decay_const=None):

        # N (number of data points) * D (number of dimensions) ndarray of buffered data stream coordinates
        self.data_buf = np.empty(shape=(0, len(features)))

        # list of feature objects
        self.features = features

        # minimum number of data points buffered for each call to fit_clusters
        self.batch_size = batch_size

        # minimum number of data points to buffer for each call to initialise_clusters
        self.init_batch_size = init_batch_size

        # number of clusters to find in data
        self.optimal_num_clusters = init_num_clusters

        # Cluster Set object storing all clusters and their attributes
        self.cluster_set = ClusterSet([])

        # class to use to define clusters 
        # ie GravitationalCluster, Cluster
        self.cluster_class = None

        # Clustering Fit classes to use in call to initialise_clusters and fit_clusters
        # ie KMeans, KMeansPlus
        self.init_clustering_class, self.clustering_class = None, None

        # Clustering Fit arguments to use in call to initialise_clusters and fit_clusters
        # ie {'n_clusters': 10, 'init': 'kmeans++'}
        self.init_clustering_class_args, self.clustering_class_args = None, None

        # initialised clustering fit object to use in call to initialise_clusters and fit_clusters
        self.init_clustering_obj, self.clustering_obj = None, None

        # number of samples passed to algorithm since initialise_clusters was last called
        self.n_samples_since_last_init = 0

        # fuzziness factor, m, =1 for hard clustering, approximately 2 for soft clustering
        self.fuzziness = fuzziness

        # maximum number of iterations allowed for each fit convergence
        self.max_iter = max_iter

        # tolerance for iteration change in centroid coordinates allowing for convergence
        self.tol = tol

        # significance constant used to calculate the optimal number of clusters
        self.alpha = alpha

        # degree to which cluster centre-of-mass should be pulled towards each new data point
        self.gravitational_const = gravitational_const

        # number counts after which a cluster should be killed
        self.time_decay_const = time_decay_const

        # weights for each point in data_buf
        self.sample_weights = np.array([])

    def calc_synthetic_inertia(self, n_samples, n_features, n_clusters, centers, std_dev):
        """
        given the centroids and standard deviation of a representative dataset, generate a
        bootstrapped datasets, cluster them to the given n_clusters and get their inertia (sse) values
        :param n_samples: number of samples to generate for the synthetic data set
        :param n_features: number of features of each data sample
        :param centers: centroids on which synthetic datasets is based
        :param std_dev: standard deviation on which synthetic dataset is based
        :param n_clusters: number of clusters to cluster synthetic dataset to
        :param sample_weights: weights to cluster synthetic datasets to
        :return: inertia associated with the bootstrapped data set
        """

        # generate synthetic datasets based on cluster parameters (mean vector and standard deviation)
        # null std_dev results in duplicated data samples at that centroid
        synthetic_dataset, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                                          cluster_std=std_dev, shuffle=False)

        # cluster the synthetic datasets
        synthetic_fit = self.fit(synthetic_dataset, n_clusters, init=centers)
        synthetic_inertia = synthetic_fit.inertia_

        return synthetic_inertia

    def generate_fit_input_data(self, data, cluster_set, weighted_centroids):
        """
        generate data points and weights as input to clusterting algorithm fit function
        :param data: buffered data points. batch_size * num_features ndarray
        :param cluster_set: ClusterSet object of all up-to-date cluster data
        :param weighted_centroids: boolean indicating whether centoirds should be weighted greater than the buffered
                                   data based on their mass
        :return data points: ndarray of data points, consisting of buffered data and existing centroids, which can be
                             passed to clustering fit call
        :return weights: ndarray of weights for data points and existing centroids, which can be passed to clustering
                         fit call
        """
        # if there are pre-existing clusters, then these are also used as data points to feed to the clustering fit
        if cluster_set.num_clusters:
            # generate data points as data buffered plus existing centroids
            del_indices = []
            for d in range(len(data)):
                if np.any(np.all(data[d] == cluster_set.centroids, axis=1), axis=0):
                    del_indices.append(d)
            reduced_data = np.delete(data, del_indices, axis=0)

            data_points = np.vstack([reduced_data, cluster_set.centroids])

            # generate equivalent weights for the new data points and the existing centroids
            if weighted_centroids:
                # clusters generated from outliers will have zero mass
                nonzero_masses = cluster_set.masses.copy()
                nonzero_masses[nonzero_masses == 0] = 1
                weights = np.concatenate([np.ones(len(reduced_data)), nonzero_masses])
            else:
                weights = np.ones(len(reduced_data) + cluster_set.num_clusters)
        # if not pre-existing clusters exist, just use the buffered data points
        else:
            data_points = data.copy()
            weights = np.ones(len(data))

        if np.any(np.isnan(weights)):
            print("here")
        return data_points, weights

    def map_clusters(self, n_old_clusters, n_new_clusters, old_centroids, new_centroids):
        # if clusters already existed, sort the list of new clusters to correspond to the old
        # and append any new to the end

        if n_old_clusters:
            old_new_centroid_distances = euclidean_distances(old_centroids, new_centroids)
            # new_old_centroid_distances = old_new_centroid_distances.T

            # for each old centroid, find index of closest corresponding new centroid,
            # the index that does not appear corresponds to a new c;ister
            old_new_cluster_set_mapping = np.argmin(old_new_centroid_distances, axis=1)
            # new_old_cluster_set_mapping = np.argmin(new_old_centroid_distances, axis=1)

            # get indices of new centroid which does not correspond to an old one
            new_cluster_idx = np.argwhere(
                ~np.isin(np.arange(n_new_clusters), old_new_cluster_set_mapping))[:, 0]

            # make a list of new clusters where their first positions correspond to the old cluster list
            uniq, uniq_index = np.unique(old_new_cluster_set_mapping, return_index=True)
            mapping = np.asarray(np.concatenate([uniq[uniq_index.argsort()], new_cluster_idx]), dtype='int')
        else:
            mapping = np.arange(n_new_clusters)

        return mapping
            # np.concatenate([new_clusters[uniq[uniq_index.argsort()]],
            #                            new_clusters[new_cluster_idx]])

    def sort_clusters(self, new_fit, new_cluster_set, mapping, n_samples):
        """

        :param new_cluster_set: ClusterSet object, NOT yet sorted
        :param new_fit: ClusteringAlgorithm fit object containing cluster_centers_, fuzzy_labels_, labels_ attributes which
                    are NOT yet ordered to correspoind to the pre-fit clustering
        :param mapping: n_new_clusters * 1 ndarray where each index i gives the index in the prefit clusters j which
                        should be located at i. This way, the approximate indices and properties of the clusters are
                        maintained through each refit and reinit
        :param n_samples: number of data points of importance. For a refit this will only include the incoming data
                          stream. For a reinit, this will also include the existing centroids.
        :return: new_clusters, new_centroids, new_labels, new_fuzzy_labels all sorted to correspond to old
        """

        # new cluster centroids
        new_centroids = new_fit.cluster_centers_[mapping]

        # re-sort cluster set
        new_cluster_set.clusters = new_cluster_set.clusters[mapping]
        new_cluster_set.centroids = new_cluster_set.centroids[mapping]
        new_cluster_set.counts = new_cluster_set.counts[mapping]
        new_cluster_set.fuzzy_counts = new_cluster_set.fuzzy_counts[mapping]
        new_cluster_set.masses = new_cluster_set.masses[mapping]
        new_cluster_set.centre_of_masses = new_cluster_set.centre_of_masses[mapping]

        # N (number of data points) * K (number of clusters) ndarray containing likliehood that each incoming data point
        # belongs to each cluster. Only consider incoming data points in case of fit_clusters,
        # or new centroids and incoming data points in case of initialise_clusters
        new_fuzzy_labels = new_fit.fuzzy_labels_[:n_samples, mapping]

        # N ndarray containing index of cluster to which each data point most likely belongs
        # resort to reflect of new index
        # get the index of each element in fit.labels_[:len(data)] in mapping to get new cluster arrangement
        # mapping.argsort() gives us the indices of the new sorted clusters at each index of the unsorted clusters
        # fit.labels_[:len(data)] gives the indices of the unsorted clusters
        new_labels = mapping.argsort()[new_fit.labels_[:n_samples]]

        return new_cluster_set, new_centroids, new_labels, new_fuzzy_labels

    def initialise_clusters(self, data, cluster_set, optimal_num_clusters, sample_count, weighted_centroids, **kwargs):
        """
        run the kmeans++ algorithm on the given data points and original cluster set centroids, where the centroids
         are weighted based on their mass if weighted_centroids=True
        :param data: batch_size or greater * num_features ndarray of buffered data points
        :param cluster_set: input ClusterSet object which stores most recent centroids, masses, counts etc of clusters
        :param optimal_num_clusters: number of clusters to fit for
        :param weighted_centroids: boolean indicating whether centroids should be weighted based on their mass for
                                   the clustering fit call
        :param kwargs: additional arguments to pass the the clustering fit call
        :return new_cluster_set: new ClusterSet object based centroids output from fit call, with count, mass and
                                 fuzzy_count values carried over from equivalent old clusters
        :return running_time: time required for fit to converge
        :return num_iters: number of iterations required for fit to converge
        :return sse: approximate total sse since last initialisation
        """

        data_points, sample_weights = self.generate_fit_input_data(data, cluster_set, weighted_centroids)

        # fit the centroids and buffered data points with kmeans++ initialisation
        running_time = datetime.now()
        init_fit = self.fit(data_points, optimal_num_clusters, weights=sample_weights, init="kmeans++")

        # get the run time, number of iterations to convergence and centroids
        running_time = (datetime.now() - running_time).total_seconds()
        num_iters = init_fit.n_iter_
        new_centroids = init_fit.cluster_centers_

        # remove data points which are now centroids
        # additional_data_points = np.empty(shape=(0, len(self.features)))
        # for d in range(len(data_points)):
        #     if not np.any([np.all(data_points[d] == new_centroids[k]) for k in range(optimal_num_clusters)]):
        #         additional_data_points = np.append(additional_data_points, data_points[d])

        # f_known = [f for f in range(len(self.features)) if
        #            self.features[f].live and self.features[f].lb is not None]
        #
        # for k in range(cluster_set.num_clusters):
        #     for f in range(len(self.features)):
        # centroids[k, f] = np.floor((self.features[f].lb +
        #                             ((self.features[f].ub - self.features[f].lb) * random())))
        # / self.features[f].step) * self.features[f].step


        mapping = self.map_clusters(cluster_set.num_clusters, optimal_num_clusters, cluster_set.centroids, new_centroids)
        new_clusters = np.array([self.cluster_class(new_centroids[k], sample_count_initialised=sample_count, **kwargs)
                                 for k in range(len(new_centroids))])
        # reinitialise new_cluster_set based on unsorted list of clusters
        new_cluster_set = ClusterSet(new_clusters)

        # re-arrange new clusters and fit results to align with old
        new_cluster_set, new_centroids, new_labels, new_fuzzy_labels = \
            self.sort_clusters(init_fit, new_cluster_set, mapping, len(data_points))

        # if old clusters exist
        # carry over normalised counts, probabilist_counts and masses
        # such that historic masses of clusters are not disregarded but also do not dominate
        # the unit mass of new data points
        if cluster_set.num_clusters:

            # get minimum nonzero values of count, probabilistic count and mass as normalisation factors for carry over
            # counts and masses will be zero for an outlier clustser, before post fit update is called
            min_count = np.min(cluster_set.counts[cluster_set.counts > 1]) \
                if len(cluster_set.counts[cluster_set.counts > 1]) else 1

            min_fuzzy_count = np.min(cluster_set.fuzzy_counts[cluster_set.fuzzy_counts > 1]) \
                if len(cluster_set.fuzzy_counts[cluster_set.fuzzy_counts > 1]) else 1

            min_mass = np.min(cluster_set.masses[cluster_set.masses > 1]) \
                if not np.isnan(cluster_set.masses[0]) and len(cluster_set.masses[cluster_set.masses > 1])  else 1

            # nonzero_variances = cluster_set.variances[np.any(cluster_set.variances > 0, axis=1)]
            # max_variance = np.max(np.sum(cluster_set.variances, axis=1))

            # loop through the new clusters and initialise their counts and masses based on the pre-existing
            # clusters if they exist
            for k in range(np.min([cluster_set.num_clusters, new_cluster_set.num_clusters])):
                # set the count to the old count, normalised
                new_cluster_set.clusters[k].count += (cluster_set.counts[k] / min_count)
                new_cluster_set.counts[k] = new_cluster_set.clusters[k].count

                # set the fuzzy_count to the old fuzzy_count, normalised
                new_cluster_set.clusters[k].fuzzy_count += \
                    (cluster_set.fuzzy_counts[k] / min_fuzzy_count)
                new_cluster_set.fuzzy_counts[k] = new_cluster_set.clusters[k].fuzzy_count

                # set the mass to the old mass, normalised
                if not np.isnan(cluster_set.masses[0]):
                    try:
                        new_cluster_set.clusters[k].mass += (cluster_set.masses[k] / min_mass)
                        new_cluster_set.masses[k] = new_cluster_set.clusters[k].mass
                    except Exception:
                        pass


        sse, compactness, dbi = self.post_fit_update(new_centroids, new_labels, new_fuzzy_labels, new_cluster_set,
                                                     data_points, sample_count, sample_weights)

        return new_cluster_set, running_time, num_iters, sse, compactness, dbi

    def fit_clusters(self, data, cluster_set, optimal_num_clusters, sample_count, weighted_centroids):
        """
        run the kmeans fit algorithm on the given data points and original cluster set centroids, where the centroids
         are weighted based on their mass if weighted_centroids=True, given the original centroids as the initialisation
        :param data: batch_size or greater * num_features ndarray of buffered data points
        :param cluster_set: input ClusterSet object which stores most recent centroids, masses, counts etc of clusters
        :param optimal_num_clusters: number of clusters to fit for
        :param weighted_centroids: boolean indicating whether centroids should be weighted based on their mass for
                                   the clustering fit call
        :param sample_count: count of data batches passed to algorithm to data
        :return new_cluster_set: new ClusterSet object based centroids output from fit call, with count, mass and
                                 fuzzy_count values carried over from equivalent old clusters
        :return running_time: time required for fit to converge
        :return num_iters: number of iterations requireed for fit to converge
        :return sse: approximate total sse since last initialisation
        """

        # get data points and weights
        data_points, sample_weights = self.generate_fit_input_data(data, cluster_set, weighted_centroids)

        # fit the centroids and buffered data points passing the centroids as initialisation instead of using kmeans++
        running_time = datetime.now()
        fit = self.fit(data_points, optimal_num_clusters, weights=sample_weights, init=cluster_set.centroids)
        new_centroids = fit.cluster_centers_

        # get the run time, number of iterations to convergence and centroids
        running_time = (datetime.now() - running_time).total_seconds()
        num_iters = self.clustering_obj.n_iter_

        mapping = self.map_clusters(cluster_set.num_clusters, optimal_num_clusters, cluster_set.centroids,
                                    new_centroids)

        new_cluster_set, new_centroids, new_labels, new_fuzzy_labels = \
            self.sort_clusters(fit, cluster_set, mapping, len(data))

        sse, compactness, dbi = self.post_fit_update(new_centroids, new_labels, new_fuzzy_labels, new_cluster_set, data,
                                                     sample_count, sample_weights)

        return cluster_set, running_time, num_iters, sse, compactness, dbi

    def post_fit_update(self, new_centroids, new_labels, new_fuzzy_labels, cluster_set, data, sample_count,
                        sample_weights):
        """
        This method is called after incoming data points have been fit to update all cluster and cluster set attributes
        :param new_centroids: n_clusters * n_features ndarray of new centroid coordinates, indices aligned with old
        :param new_labels: n_samples * 1 ndarray of indices of clusters closest to each data point, indices aligned with
         old
        :param new_fuzzy_labels: n_samples * n_clusters ndarray of membership probability of each data point to each
        cluster, indices aligned with old
        :param cluster_set: ClusterSet object
        :param data: incoming data stream samples
        :param sample_count: count of this incoming batch of data samples
        :param a mapping array where at each index is the index of the pre-fit clusters that should be here, this
               carries over pre-fit centroids to post-fit centroids with the same indices
        :return:
        """

        new_fuzzy_weights = sample_weights[:, np.newaxis] * (new_fuzzy_labels ** self.fuzziness)

        # loop through all new data points
        for d in range(len(data)):
            # loop through all existing clusters
            for k in range(len(new_centroids)):
                # update the count, probabilistic count, covariance, batch count since last update of this cluster
                # based on this data point, relative to membership probability
                # decay the mass of clusters to which no data points most likely belonged,
                # relative to fuzzy_count
                cluster_set.clusters[k].update_data(sample_count, new_data_point=data[d],
                                                    fuzzy_label=new_fuzzy_labels[d, k],
                                                    fuzzy_weight=new_fuzzy_weights[d, k],
                                                    is_closest=True if new_labels[d] == k else False,
                                                    tol=self.tol)

                # update up-to-date SSE and compactness metrics relative to membership probability
                cluster_set.clusters[k].update_metrics(sse=norm((data[d] - new_centroids[k]), 2) ** 2,
                                                       compactness=norm(data[d] - new_centroids[k], 1),
                                                       fuzzy_label=new_fuzzy_labels[d, k],
                                                       fuzzy_weight=new_fuzzy_weights[d, k])

            # update ClusterSet metrics and DBI relative to membership probability
            cluster_set.update(fuzzy_weights=new_fuzzy_weights[d, :])

        # loop through clusters
        for k in range(cluster_set.num_clusters):

            cluster_set.masses[k] = cluster_set.clusters[k].mass

            # update the centroid of the cluster
            cluster_set.clusters[k].update_centroid(new_centroid=new_centroids[k])
            cluster_set.centroids[k] = cluster_set.clusters[k].centroid

            # update the centre of mass of the cluster
            cluster_set.clusters[k].update_centre_of_mass(data, new_fuzzy_labels[:, k], self.tol)
            cluster_set.centre_of_masses[k] = cluster_set.clusters[k].centre_of_mass

        # sum the sse resulting from each new data point added since last initialisation
        sse = np.sum(cluster_set.sse_since_last_init)
        compactness = np.mean(cluster_set.cp_since_last_init)
        dbi = np.mean(cluster_set.dbi_since_last_init)

        return sse, compactness, dbi

    def feed_data(self, new_data, new_sample_weights=None):
        """
        update self.data_buf with clean new incoming data points
        :param new_data: pandas row of data
        """

        # if no data has yet been added to buffer, initialise multi-dimensional array
        # if len(self.data_buf) == 0:
        #     self.data_buf = self.data_buf.reshape(0, new_data.shape[0])

        # clean data
        # new_data = new_data.dropna(axis=0, how='any')[np.newaxis, :]
        if new_data.ndim == 1:
            new_data = new_data[np.newaxis]

        new_data = new_data[np.where(~np.any(np.isnan(new_data), axis=1))]

        self.data_buf = np.vstack([self.data_buf, new_data])

        if new_sample_weights is not None:
            new_sample_weights = new_sample_weights[np.where(~np.any(np.isnan(new_data), axis=1))]
            self.sample_weights = np.concatenate([self.sample_weights, new_sample_weights])

    def flush_data(self):
        """
        empty the data buffer
        """

        # flush data buffered from memory
        self.data_buf = np.empty(shape=(0, len(self.features)))


class OnlineKMeansPlus(ClusteringAlgorithm):
    """
    Class defining the baseline KMeansPlus Clustering Algorithm, in which the centroids are re-initialised periodically
    with the kmeans++ algorithm
    """

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size, max_iter, tol, fuzziness):
        super().__init__(features=features, init_num_clusters=init_num_clusters, batch_size=batch_size,
                         init_batch_size=init_batch_size, max_iter=max_iter, tol=tol, fuzziness=fuzziness)

        # see parent class ClusteringAlgorithm for details
        self.init_clustering_class = FuzzyKMeansPlus
        self.init_clustering_class_args = {'init': 'kmeans++', 'fuzziness': self.fuzziness, 'max_iter': self.max_iter,
                                           'tol': self.tol}  # , 'n_jobs': -1}
        self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)

        self.clustering_class = FuzzyKMeansPlus
        self.clustering_class_args = {'fuzziness': self.fuzziness, 'max_iter': self.max_iter,
                                      'tol': self.tol}
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)

        self.cluster_class = GravitationalCluster

    def fit(self, data, n_clusters, weights=None, init="kmeans++"):
        """
        fit the given data and weights for the given number of clusters, either by kmeans++ or with given initialisation
        centroids
        :param data: N * num_features ndarray of data points to cluster
        :param weights: N * 1 ndarray of weights corresponding to each data point for clustering
        :param n_clusters: number of clusters to fit for
        :param init: either "kmeans++" for such an initialisation, or a K * num_features ndarray of centroids to be
                     passed as initialisation
        :return: clustering fit object containing centroids, fuzzy_labels, hard labels, inertia (sse),
                 number of iterations
        """
        # a re-initialisation where convergence is initialised with kmeans++
        if type(init) is str and init == "kmeans++":

            # update the initialisation clustering object with the given number of clusters
            self.init_clustering_class_args.update([('n_clusters', n_clusters)])
            self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
            return self.init_clustering_obj.fit(data, sample_weights=weights)
        # a re-fitting where convergence is initialised with given old centroids
        else:
            # update the clustering object with the given number of clusters
            # and pass existing centroids as initialisation
            self.clustering_class_args.update([('n_clusters', n_clusters),
                                               ('init', init)])
            self.clustering_obj = self.clustering_class(**self.clustering_class_args)
            return self.clustering_obj.fit(data, sample_weights=weights)

    def update_clusters(self, sample_count, pool):
        """
        decide whether to initialise the clusters for the first time, to re-initialise the clusters, or to re-fit
        the clusters based on the most recent batch of incoming data
        :param sample_count: count of data batch passed to Clustering Algorithm
        :return cluster_set: ClusterSet object containing all up-to-date cluster data
        :return running_time: Running Time required for this convergence
        :return num_iters: Number of iterations required for this convergence
        :return sse: approximate SSE based on data points added since last initialisation
        """

        # declare running time, number of iterations and sse for this convergence
        running_time = np.nan
        num_iters = np.nan
        sse = np.nan
        compactness = np.nan
        dbi = np.nan

        # flags show whether clusters ought to be reinitialised or refitted
        re_init = False
        re_fit = False

        # clean nan values
        # self.data_buf = self.data_buf[~np.isnan(self.data_buf)]

        # else if clusters have not yet been initialised for the first time
        # and sufficient data points are available to fit data
        if self.cluster_set.num_clusters == 0 and len(self.data_buf) >= self.optimal_num_clusters:
            print(f"Clusters initialised for the first time. Re-initialise clusters.")
            re_init = True
            # update upper and lower bounds of features for better initialisation
            # new_ubs = np.max(self.data_buf, axis=0)
            # new_lbs = np.min(self.data_buf, axis=0)
            # for f in range(len(self.features)):
            #     if self.features[f].ub is None or new_ubs[f] > self.features[f].ub:
            #         self.features[f].ub = new_ubs[f]
            #     if self.features[f].lb is None or new_lbs[f] < self.features[f].lb:
            #         self.features[f].lb = new_lbs[f]

        # if cluster centroids have been initialised for the first time and it is time to for another
        # periodic reinitialisation
        elif self.cluster_set.num_clusters > 0 and (sample_count % self.init_batch_size) == 0:
            print(f"Number of samples since the last initialisation has exceeded initial batch size parameter"
                  f" {self.init_batch_size}. Re-initialise clusters.")
            # (re)initialise cluster means by passing data buffer and existing centroids to kmeans++ algorithm
            # with equal sample weights
            re_init = True

        # else if cluster centres must not yet be (re)initialised and sufficient data points are available
        # feed data to stream to update clusters
        elif self.cluster_set.num_clusters > 0 and len(self.data_buf) >= self.batch_size:
            print(f"There is no need to re-initialise clusters yet. Re-fit clusters.")
            re_fit = True

        # re-initialise clusters if necessary
        if re_init:
            self.cluster_set, running_time, num_iters, sse, compactness, dbi = \
                self.initialise_clusters(self.data_buf, self.cluster_set, self.optimal_num_clusters,
                                         sample_count=sample_count, weighted_centroids=False)
            self.n_samples_since_last_init = len(self.data_buf)

            # flush old data from memory
            self.flush_data()

        # otherwise re-fit clusters if possible
        elif re_fit:

            self.cluster_set, running_time, num_iters, sse, compactness, dbi = \
                self.fit_clusters(self.data_buf, self.cluster_set, self.optimal_num_clusters,
                                  sample_count=sample_count, weighted_centroids=False)
            self.n_samples_since_last_init += len(self.data_buf)

            # flush old data from memory
            self.flush_data()

        return self.cluster_set, running_time, num_iters, sse, compactness, dbi


class OnlineOptimalKMeansPlus(ClusteringAlgorithm):
    """
    Class defining OptimalKMeansPlus ClusteringAlgorithm.
    This runs 3 clustering algorithms in parallel
    for K-1, K and K+1 number of clusters, and increases or decreases K (the optimal number of clusters) by 1 if the
    DBI of the clustering solution is becoming consistently worse.
    It integrates fuzziness into the clustering.
    It applies a gravitational effect to the centre-of-masses in order to follow trends in the data not captured by the
    centroids.
    """

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size, gravitational_const=1,
                 time_decay_const=100, fuzziness=1, alpha=0.1, max_iter=10000, tol=0.00006, window_size=2):
        super().__init__(features=features, init_num_clusters=init_num_clusters, batch_size=batch_size,
                         init_batch_size=init_batch_size, fuzziness=fuzziness, max_iter=max_iter, tol=tol,
                         alpha=alpha, gravitational_const=gravitational_const, time_decay_const=time_decay_const)

        # see ClusteringAlgorithm parent class for notes

        self.init_clustering_class = FuzzyKMeansPlus
        self.init_clustering_class_args = {'init': 'kmeans++', 'fuzziness': self.fuzziness, 'max_iter': self.max_iter,
                                           'tol': self.tol}
        self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
        self.clustering_class = FuzzyKMeansPlus
        self.clustering_class_args = {'fuzziness': self.fuzziness, 'max_iter': self.max_iter,
                                      'tol': self.tol}
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)

        self.cluster_class = GravitationalCluster

        self.fuzziness = fuzziness

        # define cluster parameters for the K-1, K and K+1 cases
        self.optimal_num_clusters = [init_num_clusters - 1, init_num_clusters, init_num_clusters + 1]
        self.cluster_set = [ClusterSet([]), ClusterSet([]), ClusterSet([])]

        # booleans indicating if outlying data points should automatically be turned into clusters,
        # otherwise their only effect is their gravitational pull on existing centre-of-masses
        self.create_outliers = True

        # booleans indicating if clusters with significantly low mass should be killed,
        # otherwise their centre-of-masses will be pulled towards new data points and that cluster will ultimately
        # merge with another
        self.kill_outliers = True

        # window of convergences over which to check for continuously increasing DBI, indicating that clusters should
        # re-initialised
        self.window_size = window_size

    def calc_opt_num_clusters(self, pool):
        """
        given the cluster parameters of datasets clustered into K-1, K and K+1 clusters,
        calculate the optimal number of clusters by generating bootstrap synthetic data sets, calculating
        the pdf of their SSE values and calculating the cd of these pdf functions up to the SSE achieved for the
        next greatest number of clusters
        :param pool: multiprocessing pool object
        :return new_optimal_num_clusters most optimal number of clusters for this dataset,
        either equal to original or +/- 1
        """

        # number of features, samples to use for synthetic datasets
        n_features = len(self.features)

        # assume first that optimal number of clusters is current value +1, if not it will be reset in the following loop
        new_optimal_num_clusters = self.optimal_num_clusters[-1]

        # loop through the already clustered datasets for K-1 and K, cluster synthetic datasets based on these values,
        # compare the resulting synthetic sse values to the sse value for the already clustered K+1 cluster solution
        for c in range(2):

            # K, number of clusters with which to cluster synthetic datasets
            k_n_clusters = self.optimal_num_clusters[c]

            # if there only exists one cluster here, then this is not a realistic value to reset K to, skip it
            if k_n_clusters == 1:
                continue

            # get the already calculated centroids and standard deviation for clustering at K number of clusters
            k_centroids = self.cluster_set[c].centroids
            # k_std_dev = [np.sqrt(np.diag(cov)) for cov in self.cluster_set[c].covariances]
            # square root of variance accrued since last initialisation
            k_std_dev = np.sqrt(self.cluster_set[c].variances_since_last_init)

            # based on the centroids and standard deviations of the actual dataset clustered with K number of clusters,
            # generate synthetic data sets, cluster them with K number of clusters
            # and store their sse values
            # synthetic_sses = np.array([])
            # for s in range(num_syn_datasets):
            #     synthetic_sses = np.append(synthetic_sses,
            #                                calc_synthetic_inertia(
            #                                    self, n_samples, n_features, k_centroids,
            #                                    k_std_dev, n_clusters, weights))

            # data samples processed since last initialisation
            # k_n_samples = self.n_samples_since_last_init #(np.sum(self.cluster_set[c].fuzzy_counts))

            # data samples assigned to each clusters
            n_samples_per_cluster = np.asarray(self.cluster_set[c].fuzzy_counts_since_last_init, dtype='int')
            k_n_samples = int(np.sum(n_samples_per_cluster))
            # number of synthetic datasets to generate
            num_syn_datasets = 10
            # number of bins to form sse histogram
            num_bins = 100

            # n_samples_per_cluster = np.zeros(n_clusters)
            # # clusters with zero standard deviation get only one sample (the centroid)
            # zero_std_dev_indices = np.sum(k_std_dev, axis=1) == 0
            # n_samples_per_cluster[zero_std_dev_indices] = 1

            # n_samples_per_cluster[~zero_std_dev_indices] = \
            #     [int((k_n_samples - np.sum(zero_std_dev_indices)) // np.sum(~zero_std_dev_indices))] \
            #     * np.sum(~zero_std_dev_indices)
            synthetic_sses = pool.starmap(calc_synthetic_inertia,
                                          [(self, n_samples_per_cluster, n_features, k_n_clusters, k_centroids, k_std_dev)
                                           for s in range(num_syn_datasets)])

            # get the sse value of already clustered dataset for K+1 number clusters
            # as this number reflects only the addition of data points to clusters as they arrive, and not how
            # these data points would later be reassigned to reduce sse, this value is overestimated, thus the median
            # is used to reduce it to a more reasonable value

            # sse accrued since last initialisation by K+1 algorithm run
            kplus_sse = self.cluster_set[c + 1].sse_since_last_init[-1]

            # calculate pdf of SSE values achieved over all of these synthetic data sets
            pdf, bin_edges = np.histogram(synthetic_sses, range=(0, np.max(synthetic_sses)),
                                          bins=num_bins, density=True)

            # pdf must be integrated over interval to yield pmf
            pmf = pdf * np.diff(bin_edges)

            # calculate cdf
            cdf = np.cumsum(pmf)

            # get p-value = cdf of SSE pdf at the SSE achieved for the next greatest number of clusters K+1
            # p value = 1 means that it is certain that the value of the sse for K
            #  (from bootstrapped datasets) is less than the actual sse for K+1, so we should stop at K

            # p value = 0 means that it is certain that the value of the sse for K
            #   (from bootstrapped datasets) is greater than the actual sse for K+1, so we should increase to K+1

            p_value = np.interp(kplus_sse, bin_edges[0:-1], cdf)

            # if there is a sufficiently high probability (approx 5-10%)
            # that the SSE resulting from clustering at K based on synthetic datasets
            # is less than SSE resulting from clustering at K+1, then update the optimal number of clusters to K and
            # break the loop
            if p_value > self.alpha:
                new_optimal_num_clusters = self.optimal_num_clusters[c]
                break

        return new_optimal_num_clusters

    def kill_outlier_cluster(self, cluster_set, outlier_idx):
        """
        kill a cluster that has not received a most likely data point since self.time_decay sample counts have passed
        :param cluster_set: original ClusterSet object
        :param outlier_data_idx: index of data sample in self.data_buf
        :return:
        """
        # copy original cluster_set
        new_cluster_set = deepcopy(cluster_set)

        # calculate euclidean distances between all centroids and the outlier centroid
        live_centroid_dead_centroid_distances = euclidean_distances(cluster_set.centroids,
                                                                    cluster_set.centroids[np.newaxis, outlier_idx])
        live_centroid_dead_centroid_distances = np.delete(live_centroid_dead_centroid_distances, outlier_idx)

        # generate coefficients between 0 and 1 which indicate what proportion of the metrics
        # (count, mass, fuzzy_count) of the dead cluster will be passed on to the other clusters
        a = 1 - (live_centroid_dead_centroid_distances / np.max(live_centroid_dead_centroid_distances))
        with np.errstate(divide='ignore', invalid='ignore'):
            a /= np.sum(a)
        a[np.isnan(a)] = 1

        # kill the outlier cluster
        new_cluster_set.kill_clusters([outlier_idx])

        print(f"Outlier cluster killed.")

        # update original clusters
        for k in range(new_cluster_set.num_clusters):
            new_cluster_set.clusters[k].count += a[k] * cluster_set.counts[outlier_idx]
            new_cluster_set.counts[k] = new_cluster_set.clusters[k].count

            new_cluster_set.clusters[k].fuzzy_count += a[k] * cluster_set.fuzzy_counts[outlier_idx]
            new_cluster_set.fuzzy_counts[k] = new_cluster_set.clusters[k].fuzzy_count

            new_cluster_set.clusters[k].mass += a[k] * cluster_set.masses[outlier_idx]
            new_cluster_set.masses[k] = new_cluster_set.clusters[k].mass

        return new_cluster_set

    def create_outlier_cluster(self, cluster_set, outlier_idx):
        """
        create a cluster from a given outlier
        :param cluster_set: original ClusterSet object
        :param outlier_data_idx: index of data sample in self.data_buf
        :return:
        """
        # copy original cluster_set
        new_cluster_set = deepcopy(cluster_set)

        # create the new clusters
        outlier_cluster = self.cluster_class(centroid=self.data_buf[outlier_idx].copy(),
                                             gravitational_const=self.gravitational_const,
                                             time_decay_const=self.time_decay_const)

        # calculate euclidean distances between all centroids and the outlier centroid
        old_centroid_new_centroid_distances = euclidean_distances(cluster_set.centroids,
                                                                  self.data_buf[outlier_idx, np.newaxis])

        # generate coefficients between 0 and 1 which indicate what proportion of the metrics
        # (count, mass, fuzzy_count) of each old cluster will be passed on to this new cluster
        a = (old_centroid_new_centroid_distances / np.max(old_centroid_new_centroid_distances)).squeeze(axis=1)

        outlier_cluster.count = np.sum((1 - a) * cluster_set.counts)
        outlier_cluster.fuzzy_count = np.sum((1 - a) * cluster_set.fuzzy_counts)
        outlier_cluster.mass = np.sum((1 - a) * cluster_set.masses)

        new_cluster_set.create_clusters([outlier_cluster])

        print(f"Outlier cluster created.")

        # update original clusters
        for k in range(cluster_set.num_clusters):
            new_cluster_set.clusters[k].count = a[k] * cluster_set.counts[k]
            new_cluster_set.counts[k] = new_cluster_set.clusters[k].count

            new_cluster_set.clusters[k].fuzzy_count = a[k] * cluster_set.fuzzy_counts[k]
            new_cluster_set.fuzzy_counts[k] = new_cluster_set.clusters[k].fuzzy_count

            new_cluster_set.clusters[k].mass = a[k] * cluster_set.masses[k]
            new_cluster_set.masses[k] = new_cluster_set.clusters[k].mass

        return new_cluster_set

    def split_cluster(self, cluster_set, sample_count):
        """
        Find the cluster with the greatest centre-of-mass/centroid euclidean difference and split it.
        :param cluster_set: original ClusterSet object
        :return: new ClusterSet object once cluster has been split
        """

        # copy original cluster_set
        new_cluster_set = deepcopy(cluster_set)

        # for no gravitational effect, store the index of the sample closest to the split cluster centroid, as the
        # second centroid
        close_point_idx = None

        if self.gravitational_const:
            # calculate euclidean distances between all centroids and centre_of_masses
            centroid_com_distances = \
                paired_distances(cluster_set.centroids, cluster_set.centre_of_masses)

            # get index of cluster with largest centroid - centre-of-mass difference
            split_idx = np.argmax(centroid_com_distances)

            # set new centroids to centroid and centre-of-mass of cluster to split
            new_centroids = [cluster_set.centroids[split_idx], cluster_set.centre_of_masses[split_idx]]
        else:
            # get index of cluster with largest variance
            cluster_std_devs = np.sum(cluster_set.variances_since_last_init, axis=1) ** 0.5
            split_idx = np.argmax(cluster_std_devs, axis=0)

            # get index of new data point closest to it
            point_centroid_distances = euclidean_distances(self.data_buf, cluster_set.centroids[split_idx, np.newaxis])

            # get index of minimum distance between new data points and their closest centroids
            close_point_idx = np.argmin(point_centroid_distances, axis=0)

            # set new centroids to centroid and centroid + std_dev of cluster to split
            new_centroids = [cluster_set.centroids[split_idx], self.data_buf[close_point_idx]]

        print(f"Cluster {split_idx} split.")

        # kill the old clusters
        new_cluster_set.kill_clusters([split_idx])

        # create the new clusters
        new_cluster_set.create_clusters([self.cluster_class(centroid=new_centroids[idx],
                                                            gravitational_const=self.gravitational_const,
                                                            time_decay_const=self.time_decay_const)
                                         for idx in range(len(new_centroids))])

        # half the counts, fuzzy_counts and masses and add to the new split clusters
        for k in [-2, -1]:
            new_cluster_set.clusters[k].count += cluster_set.counts[split_idx] / 2
            new_cluster_set.counts[k] = new_cluster_set.clusters[k].count

            new_cluster_set.clusters[k].fuzzy_count += cluster_set.fuzzy_counts[split_idx] / 2
            new_cluster_set.fuzzy_counts[k] = new_cluster_set.clusters[k].fuzzy_count

            new_cluster_set.clusters[k].mass += cluster_set.masses[split_idx] / 2
            new_cluster_set.masses[k] = new_cluster_set.clusters[k].mass

        return new_cluster_set

    def merge_clusters(self, cluster_set, merge_indices, sample_count):

        # copy original cluster_set
        new_cluster_set = deepcopy(cluster_set)

        # the new centroid is the weighted mean of the merging clusters
        new_centroid = np.mean(cluster_set.masses[merge_indices]
                               * cluster_set.centroids[merge_indices, :] /
                               np.sum(cluster_set.masses[merge_indices]), axis=0)

        # kill the original clusters
        new_cluster_set.kill_clusters(merge_indices)

        # create the new cluster
        new_cluster_set.create_clusters([
            self.cluster_class(centroid=new_centroid, gravitational_const=self.gravitational_const,
                               time_decay_const=self.time_decay_const,
                               sample_count_initialised=sample_count)])

        # sum their counts, fuzzy_counts and masses and add to the new merged cluster
        new_cluster_set.clusters[-1].count += np.sum(cluster_set.counts[merge_indices])
        new_cluster_set.counts[-1] = new_cluster_set.clusters[-1].count

        new_cluster_set.clusters[-1].fuzzy_count += \
            np.sum(cluster_set.fuzzy_counts[merge_indices])
        new_cluster_set.fuzzy_counts[-1] = new_cluster_set.clusters[-1].fuzzy_count

        new_cluster_set.clusters[-1].mass += np.sum(cluster_set.masses[merge_indices])
        new_cluster_set.masses[-1] = new_cluster_set.clusters[-1].mass

        print(f"Clusters {merge_indices[0]} and {merge_indices[1]} merged.")

        return new_cluster_set

    def reduce_clusters(self, cluster_set, sample_count):
        """
        find the cluster with the lowest mass and kill it
        or find the pair of clusters with the lowest centre-of-mass euclidean difference and merge them
        :param cluster_set: original cluster set object
        :return: new cluster set object with merged or killed clusters
        """

        # copy original cluster set
        new_cluster_set = deepcopy(cluster_set)

        # get zscore of cluster masses
        mass_z = stats.zscore(cluster_set.masses, axis=None)

        # get index of cluster with lowest masses
        mass_outlier_index = np.argmin(mass_z)

        # if the cluster with the lowest mass was created before the last initialisation
        # such that it has had time to grow before being killed
        # kill_condition = True
        # self.kill_outliers \
        #              and (cluster_set.clusters[mass_outlier_index].sample_count_initialised
        #                   < sample_count - self.n_samples_since_last_init)

        # if there are not enough clusters for zscore comparison between inter-centroid distances (<=2)
        # and we are allowed to, kill the cluster with the lowest mass
        if cluster_set.num_clusters <= 2:
            # if kill_condition:
            new_cluster_set = self.kill_outlier_cluster(cluster_set, mass_outlier_index)
            # else:
            #     print(f"Cluster number not reduced as clusters to kill have been only recently created.")
        # else if there are enough clusters for zscore comparision between inter-centroid distances
        else:

            # calculate distances separating centroids and isolate unique nonzero differences
            tril_indices = np.tril_indices(cluster_set.num_clusters, k=-1)
            centroid_distances = pairwise_distances(cluster_set.centroids)[tril_indices]

            # get zscore of distances separating centroids
            centroid_z = stats.zscore(centroid_distances, axis=None)
            centroid_z_outlier_index = np.argmin(centroid_z)

            # get index of clusters with lowest centroid separation
            centroid_outlier_index = np.array([tril_indices[0][centroid_z_outlier_index],
                                               tril_indices[1][centroid_z_outlier_index]])

            # if the z-value associated with the lightest cluster is less (more negative) than that
            # associated with the inter-centroid distances
            if mass_z[mass_outlier_index] < centroid_z[centroid_z_outlier_index]:
                # if we are allowed kill the cluster with the lowest mass
                # if kill_condition:
                # kill the cluster
                new_cluster_set = self.kill_outlier_cluster(cluster_set, mass_outlier_index)
                # else:
                #     print(f"Cluster number not reduced as clusters to kill have been only recently created.")
            # else if the z-value associated with the lightest cluster is greater (less negative) than that
            # associated with the inter-centroid distances
            else:
                # merge the clusters with close centroids
                new_cluster_set = self.merge_clusters(cluster_set, centroid_outlier_index, sample_count)


        return new_cluster_set

    def update_clusters(self, sample_count, pool):
        """
        decide whether to initialise the clusters for the first time, to re-initialise the clusters, or to re-fit
        the clusters based on the most recent batch of incoming data
        :param sample_count: count of data batch passed to Clustering Algorithm
        :return cluster_set: ClusterSet object containing all up-to-date cluster data
        :return running_time: Running Time required for this convergence
        :return num_iters: Number of iterations required for this convergence
        :return sse: approximate SSE based on data points added since last initialisation
        """

        # declare running time, number of iterations and sse variables
        running_time = np.empty(3)
        running_time[:] = np.nan
        num_iters = np.empty(3)
        num_iters[:] = np.nan
        sse = np.empty(3)
        sse[:] = np.nan
        compactness = np.empty(3)
        compactness[:] = np.nan
        dbi = np.empty(3)
        dbi[:] = np.nan

        # three clustering algorithms are run in parallel for K-1,K and K+1 number of clusters, where the middle index
        # at 1 represents the current optimal K number of clusters
        opt_index = 1

        # flags show whether clusters ought to be reinitialised or refitted
        re_init = False
        re_fit = False

        # order of DBI differenct to check for re-initialisation condition
        diff_order = 2

        # check if clusters have not yet been initialised and
        # if enough data points have been buffered to initialise clusters for the first time
        if self.cluster_set[opt_index].num_clusters == 0 and len(self.data_buf) >= self.optimal_num_clusters[-1]:
            print(f"Clusters initialised for the first time. Re-initialise clusters.")
            re_init = True

        # else if the clusters have been initialised for the first time,
        # check if the clusters ought to be re-initialised based on increasing rate of increase in cluster DBI
        # n_samples_since_last_init is used to generate bootstrapped data sets which are clustered for K-1, K and K+1
        # values, so it must be at least as great as the last element of self.optimal_num_clusters (K+1)

        elif self.n_samples_since_last_init >= self.optimal_num_clusters[-1] \
            and (self.n_samples_since_last_init >= self.init_batch_size or
                (self.n_samples_since_last_init > self.window_size + diff_order
                 and np.all(np.diff(self.cluster_set[opt_index].dbi_since_last_init, n=diff_order)[-self.window_size:]
                            > 0))):

            if self.n_samples_since_last_init >= self.init_batch_size:
                print(f"Number of samples since the last initialisation has exceeded initial batch size parameter"
                      f" {self.init_batch_size}.")
            else:
                print(f"DBI has had a positive rate of increase for {self.window_size} samples.")

            # calculate optimal number of clusters
            new_optimal_num_clusters = self.calc_opt_num_clusters(pool)

            # if the optimal number of clusters is greater than the current number of clusters,
            # split the c
            # luster with the highest centroid, centre-of-mass difference and re-initialise based
            # on these centroids by KMeans++

            if new_optimal_num_clusters >= 2:
                old_optimal_num_clusters = self.optimal_num_clusters[opt_index]

                # if the optimal number of clusters has increased by 1, expand clusters and re-initialise the clusters
                if new_optimal_num_clusters > old_optimal_num_clusters:
                    # every time Kopt is increased,
                    # make it exponentially less likely that it will be increased again by lowering the threshold
                    # for the pvalue.
                    # The greater the current number of clusters, the less alpha is decreased,
                    # the less likely K will increase again

                    # self.alpha *= np.exp(-(1 / (new_optimal_num_clusters)))
                    print(f"New optimal number of clusters has increased."
                          f" Expand cluster by splitting, then re-initialise clusters.")
                    for c in range(3):
                        # the clusters are re-initialised (counts, sse, centroids and masses are reset)
                        # self.cluster_set[c], running_time[c], num_iters[c], sse[c] = \
                        self.cluster_set[c] = self.split_cluster(cluster_set=self.cluster_set[c],
                                                                  sample_count=sample_count)

                    re_init = True

                # else if the optimal number of clusters has decreased by 1, reduce clusters and
                # re-initialise the clusters
                elif new_optimal_num_clusters < old_optimal_num_clusters:
                    # every time Kopt is decreased,
                    # make it exponentially less likely that it will be decreased again by increasing the threshold
                    # for the pvalue.
                    # The less the current number of clusters, the more alpha is increased,
                    # the less likely K will decrease again

                    # self.alpha *= np.exp(1 / (new_optimal_num_clusters))
                    print(f"New optimal number of clusters has decreased."
                          f" Reduce clusters by merging or killing, then re-initialise clusters.")
                    for c in range(3):
                        self.cluster_set[c] = self.reduce_clusters(self.cluster_set[c], sample_count)
                    re_init = True

                # else if the optimal number of clusters has not changed, re-fit the clusters
                elif new_optimal_num_clusters == old_optimal_num_clusters:
                    print(f"New optimal number of clusters has not changed."
                          f" Re-initialise clusters.")
                    re_init = True

                # update optimal num clusters list to reflect what increases/decreases in cluster number could actually
                # be executed in the code reduce/split_cluster functions
                new_optimal_num_clusters = self.cluster_set[opt_index].num_clusters
                self.optimal_num_clusters = [new_optimal_num_clusters - 1,
                                             new_optimal_num_clusters,
                                             new_optimal_num_clusters + 1]

        # else if cluster centres have already been initialised for the first time,
        # must not yet be (re)initialised, and sufficient data points are available,
        # feed data to stream to update clusters
        elif self.cluster_set[opt_index].num_clusters > 0 and len(self.data_buf) >= self.batch_size:

            # check if the minimum distance from any of the new data points to their closest centroid is greater than
            # any inter centroidal distance. If so, make a new outlying cluster, which will decay over time and be
            # killed if redundant

            # check distances between new points and closest existing centroids
            point_centroid_distances = euclidean_distances(self.data_buf, self.cluster_set[opt_index].centroids)
            min_point_centroid_distance = np.min(point_centroid_distances, axis=1)

            # get index of maximum distance between new data points and their closest centroids
            outlier_idx = np.argmax(min_point_centroid_distance)

            # check distances between existing centroids
            inter_centroid_distances = euclidean_distances(self.cluster_set[opt_index].centroids)
            #inter_centroid_distances = inter_centroid_distances[np.where(inter_centroid_distances != 0)]

            # get index of maximum distance between any two centroids
            max_inter_centroid_distance = np.max(inter_centroid_distances)

            # if the distance from the most outlying data point to its closest centroid is greater than any
            # inter-centroidal distance, then create a new cluster from the outlier
            if min_point_centroid_distance[outlier_idx] > max_inter_centroid_distance and self.create_outliers:

                # increment K
                self.optimal_num_clusters = [n + 1 for n in self.optimal_num_clusters]

                # create new clusters in each of the three parallel algorithm runs for K-1, K and K+1
                for c in range(3):
                    self.cluster_set[c] = self.create_outlier_cluster(self.cluster_set[c], outlier_idx)

                re_init = True

            # get index of cluster with the smallest mass
            negligible_mass_indices = np.argwhere(self.cluster_set[opt_index].masses < self.tol)
            # if the smallest cluster mass is less than the allowed tolerance
            for negligible_mass in negligible_mass_indices:
                # decrement K
                self.optimal_num_clusters = [n - 1 for n in self.optimal_num_clusters]

                # kill or merge new clusters in each of the three parallel algorithm runs for K-1 and K+1
                for c in range(3):
                    self.cluster_set[c] = self.reduce_clusters(self.cluster_set[c], sample_count)

                # kill new clusters in parallel algorithm run for K
                # outlier_idx = np.argmin(self.cluster_set[c].masses)
                # self.cluster_set[1] = self.kill_outlier_cluster(self.cluster_set[c], outlier_idx)

                re_init = True

            # get indices of centroids which have converged on eachother
            # get index of 0 distance between any two centroids
            inter_centroid_distances = euclidean_distances(self.cluster_set[opt_index].centroids)
            tril_indices = np.tril_indices(self.cluster_set[opt_index].num_clusters, k=-1)
            converged_indices = np.argwhere(inter_centroid_distances < self.tol)
            converged_indices = [idx for idx in converged_indices
                                 if idx[0] in tril_indices[0] and idx[1] in tril_indices[1] and idx[0] != idx[1]]
            converged_indices = np.unique(converged_indices)
            # if the smallest cluster mass is less than the allowed tolerance
            for converged_idx in converged_indices:
                # decrement K
                self.optimal_num_clusters = [n - 1 for n in self.optimal_num_clusters]

                # kill or merge new clusters in each of the three parallel algorithm runs for K-1, K and K+1
                for c in range(3):
                    self.cluster_set[c] = self.reduce_clusters(self.cluster_set[c], sample_count)

                # merge new clusters in parallel algorithm run for K
                # self.cluster_set[opt_index] = self.merge_clusters(self.cluster_set[opt_index], converged_idx, sample_count)

                re_init = True


            if not re_init:
                print(f"No outliers have been found and there is no need to re-initialise clusters yet."
                      f" Re-fit clusters.")
                re_fit = True

        # re-initialise clusters if necessary for each of the algorithm runs
        if re_init:

            fit_results = pool.starmap(initialise_clusters,
                                        [(self, {'data': self.data_buf, 'cluster_set': self.cluster_set[c],
                                                 'optimal_num_clusters': self.optimal_num_clusters[c],
                                                 'sample_count': sample_count,
                                                 'weighted_centroids': True,
                                                 'gravitational_const': self.gravitational_const,
                                                 'time_decay_const': self.time_decay_const}) for c in
                                         range(3)])

            self.n_samples_since_last_init = len(self.data_buf)

            # fit_results = [[], [], []]
            # for c in range(3):
            #      res = initialise_clusters(self, {'data': self.data_buf, 'cluster_set': self.cluster_set[c],
            #                                    'optimal_num_clusters': self.optimal_num_clusters[c],
            #                                    'sample_count': sample_count,
            #                                    'weighted_centroids': True,
            #                                    'gravitational_const': self.gravitational_const})
            #      fit_results[c] = res

        # otherwise re-fit clusters if possible for each of the parallel algorithm runs
        elif re_fit:

            fit_results = pool.starmap(fit_clusters,
                                       [(self, {'data': self.data_buf, 'cluster_set': self.cluster_set[c],
                                                'optimal_num_clusters': self.optimal_num_clusters[c],
                                                'weighted_centroids': True,
                                                'sample_count': sample_count})
                                        for c in range(3)])

            self.n_samples_since_last_init += len(self.data_buf)

        if re_init or re_fit:

            self.cluster_set = [res[0] for res in fit_results]
            running_time = [res[1] for res in fit_results]
            num_iters = [res[2] for res in fit_results]
            sse = [res[3] for res in fit_results]
            compactness = [res[4] for res in fit_results]
            dbi = [res[5] for res in fit_results]

            # for c in range(3):
            #     self.cluster_set[c], running_time[c], num_iters[c], sse[c], compactness[c], dbi[c] = \
            #         fit_clusters(self, {'data': self.data_buf, 'cluster_set': self.cluster_set[c],
            #                             'optimal_num_clusters': self.optimal_num_clusters[c],
            #                             'weighted_centroids': True,
            #                             'sample_count': sample_count})


            # flush old data from memory if it was used to fit clustering
            self.flush_data()

        return self.cluster_set[opt_index], running_time[opt_index], num_iters[opt_index], sse[opt_index], \
               compactness[opt_index], dbi[opt_index]

    def fit(self, data, optimal_num_clusters, weights=None, init="kmeans++"):
        """
        fit the given data and weights for the given number of clusters, either by kmeans++ or with given initialisation
        centroids
        :param data: N * num_features ndarray of data points to cluster
        :param weights: N * 1 ndarray of weights corresponding to each data point for clustering
        :param optimal_num_clusters: number of clusters to fit for
        :param init: either "kmeans++" for such an initialisation, or a K * num_features ndarray of centroids to be
                     passed as initialisation
        :return: clustering fit object containing centroids, fuzzy_labels, hard labels, inertia (sse),
                 number of iterations
        """

        # a re-initialisation where convergence is initialised with kmeans++
        if (type(init) is str) and (init == "kmeans++"):

            # update the initialisation clustering object with the given number of clusters with kmeans++ algorithm
            # given the existing centroids and incoming data points as a basis, where the existing centroids have a
            # weight proportional to their mass
            self.init_clustering_class_args.update([('n_clusters', optimal_num_clusters)])
            self.init_clustering_obj = self.init_clustering_class(**self.init_clustering_class_args)
            return self.init_clustering_obj.fit(data, sample_weights=weights)
        # a re-fitting where convergence is initialised with given old centroids
        else:
            # update the clustering object with a new number of clusters and pass existing centroids as initialisation
            self.clustering_class_args.update([('n_clusters', optimal_num_clusters),
                                               ('init', init)])
            self.clustering_obj = self.clustering_class(**self.clustering_class_args)
            return self.clustering_obj.fit(data, sample_weights=weights)


# class OfflineOPTICS(OfflineOptimalKMeansPlus):
#     def __init__(self, features, init_num_clusters, batch_size, init_batch_size, fuzziness=2,
#                  max_iter=10000, tol=0.000006, alpha=None, gravitational_const=None, time_decay_const=None,
#                  cluster_set=None):
#
#         super().__init__(features=features, init_num_clusters=init_num_clusters, batch_size=batch_size,
#                          init_batch_size=init_batch_size, fuzziness=fuzziness, max_iter=max_iter, tol=tol, alpha=alpha,
#                          gravitational_const=gravitational_const, time_decay_const=time_decay_const,
#                          cluster_set=cluster_set)
#
#         self.clustering_class = OPTICS
#         self.clustering_class_args = {'min_samples': 2, 'metric': 'euclidean'}
#         self.clustering_obj = self.clustering_class(**self.clustering_class_args)
#
#     def fit_clusters(self, data, pool=None):
#         # cluster_set = ClusterSet([self.cluster_class(centroid=centroids[k], sample_count_initialised=None)
#         #                           for k in range(n_clusters)])
#         # sse, compactness, dbi = self.post_fit_update(k_fit, cluster_set, self.data_buf, 0, sample_weights,
#         #                                              np.arange(cluster_set.num_clusters))
#         pass
#
#     def fit(self, data):
#         """
#         fit the given data and weights for the given number of clusters, either by kmeans++ or with given initialisation
#         centroids
#         :param data: N * num_features ndarray of data points to cluster
#         :param weights: N * 1 ndarray of weights corresponding to each data point for clustering
#         :param optimal_num_clusters: number of clusters to fit for
#         :param init: either "kmeans++" for such an initialisation, or a K * num_features ndarray of centroids to be
#                      passed as initialisation
#         :return: clustering fit object containing centroids, fuzzy_labels, hard labels, inertia (sse),
#                  number of iterations
#         """
#
#         # convergence is initialised with kmeans++
#         # update the clustering object with the given number of clusters with kmeans++ algorithm
#         # given the total dataset
#         return self.clustering_obj.fit(data)


class OfflineOptimalKMeansPlus(ClusteringAlgorithm):
    """
    Class defining Offline OptimalKMeansPlus ClusteringAlgorithm as tuning protocol and baseline to online version.
    """

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size, fuzziness=2,
                 max_iter=10000, tol=0.000006, alpha=None, gravitational_const=None, time_decay_const=None,
                 cluster_set=None):

        super().__init__(features=features, init_num_clusters=init_num_clusters, batch_size=batch_size,
                         init_batch_size=init_batch_size, fuzziness=fuzziness, max_iter=max_iter, tol=tol, alpha=alpha,
                         gravitational_const=gravitational_const, time_decay_const=time_decay_const)

        # see notes from parent class ClusteringAlgorithm

        self.clustering_class = FuzzyKMeansPlus
        self.clustering_class_args = {'init': 'kmeans++', 'fuzziness': self.fuzziness, 'max_iter': self.max_iter,
                                      'tol': self.tol}
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)

        self.cluster_class = GravitationalCluster

        self.optimal_num_clusters = init_num_clusters

        self.cluster_set = ClusterSet([]) if cluster_set is None else cluster_set

    def update_clusters(self, pool):
        # clean data
        # data = self.data_buf.loc[[feat.name for feat in self.features]].dropna().values

        # fit data
        return self.fit_clusters(self.data_buf, pool)

    def fit_clusters(self, data, pool):
        """
        run the kmeans fit algorithm on the given data points and original cluster set centroids, where the centroids
         are weighted based on their mass if weighted_centroids=True, given the original centroids as the initialisation
        :param data: batch_size or greater * num_features ndarray of buffered data points
        :return new_cluster_set: new ClusterSet object based centroids output from fit call, with count, mass and
                                 fuzzy_count values carried over from equivalent old clusters
        :return running_time: time required for fit to converge
        :return num_iters: number of iterations requireed for fit to converge
        :return sse: total inertia (sse) since last initialisation
        """

        n_samples, n_features = data.shape

        # number of synthetic datasets to generate
        num_syn_datasets = 100

        # number of bins to form sse histogram
        num_bins = 1000

        # weights for each sample
        sample_weights = self.sample_weights

        running_time = datetime.now()

        # loop through possible values of K (optimal number of clusters)
        old_n_clusters = self.cluster_set.num_clusters
        old_centroids = self.cluster_set.centroids
        new_n_clusters = old_n_clusters
        new_centroids = old_centroids.copy()

        new_fit = self.fit(data, 2, sample_weights)

        for c in range(2, n_samples):
            print(f"\nTesting n_clusters = {c}.")
            # fit the actual data with this many clusters
            k_n_clusters = c
            k_fit = self.fit(data, k_n_clusters, sample_weights)
            k_labels = k_fit.labels_
            k_fuzzy_labels = k_fit.fuzzy_labels_
            k_centroids = k_fit.cluster_centers_
            _, k_n_samples = np.unique(k_labels, return_counts=True)
            k_n_probabilistic_samples = np.sum(k_fuzzy_labels, axis=0)

            k_std_dev = np.zeros(shape=(k_n_clusters, n_features))
            # k_std_dev = np.zeros(shape=(n_clusters, n_features))
            for k in range(k_n_clusters):
                if k_n_probabilistic_samples[k] > 1:
                    k_std_dev[k] = np.sqrt((1 / (k_n_probabilistic_samples[k] - 1))
                                           * np.sum(k_fuzzy_labels[:, k, np.newaxis]**self.fuzziness
                                                 * (data - k_centroids[k]) ** 2, axis=0))
                    # (np.tile(k_fuzzy_labels[:, k], (n_features, 1)).T)
                    # if k_n_samples[k] > 1:
                    #     k_std_dev[k] = np.sqrt((1 / (k_n_samples[k] - 1))
                    #                            * np.sum((data[k_labels == k] - k_centroids[k]) ** 2, axis=0))
                    # [k_labels == k]

            # generate synthetic datasets based on the centroids and standard deviations
            # of actual dataset clustered with K clusters
            # cluster the synthetic datasets with K clusters
            # fetch the resulting sse values
            print(f"Calculating synthetic SSE values for n_clusters = {c}.")

            synthetic_sses = pool.starmap(calc_synthetic_inertia,
                                          [(self, np.asarray(k_n_probabilistic_samples, dtype='int'), n_features,
                                            k_n_clusters, k_centroids, k_std_dev)
                                           for s in range(num_syn_datasets)])

            # sse values from clustered synthetic datasets
            # synthetic_sses = []
            #
            # for s in range(num_syn_datasets):
            #     synthetic_sses.append(
            #         calc_synthetic_inertia(self, k_n_samples, n_features, k_centroids, k_std_dev, n_clusters, weights))

            # fit the actual dataset with K+1 clusters and get the resulting sse
            kplus_n_clusters = c + 1
            kplus_fit = self.fit(data, kplus_n_clusters, sample_weights=sample_weights)
            kplus_sse = kplus_fit.inertia_

            # calculate the pdf of the synthetic sse values
            pdf, bin_edges = np.histogram(synthetic_sses, range=(0, np.max(synthetic_sses)), bins=num_bins,
                                          density=True)

            pmf = pdf * np.diff(bin_edges)

            # calculate the cdf of the synthetic sse values
            cdf = np.cumsum(pmf)

            # calculate the p-value = value of cdf of synthetic sse values at the sse value of the actual K+1 clustering
            # = probability of finding a synthetic K-clustered dataset with an SSE lower than the sse of the actula
            # K+1-clustered dataset
            p_value = np.interp(kplus_sse, bin_edges[0:-1], cdf)

            # if the p-value is significant, then we choose K as the optimal number of clusters
            if p_value > self.alpha or c == n_samples - 1:
                new_fit = k_fit
                new_n_clusters = k_n_clusters
                new_centroids = k_centroids
                break


        # get the running time
        running_time = (datetime.now() - running_time).total_seconds()

        # create the new ClusterSet
        mapping = self.map_clusters(old_n_clusters, new_n_clusters, old_centroids, new_centroids)
        kwargs = {}
        new_clusters = np.array([self.cluster_class(new_centroids[k], sample_count_initialised=None, **kwargs)
                                 for k in range(len(new_centroids))])
        # reinitialise new_cluster_set based on unsorted list of clusters
        new_cluster_set = ClusterSet(new_clusters)

        # re-arrange new clusters and fit results to align with old
        new_cluster_set, new_centroids, new_labels, new_fuzzy_labels = \
            self.sort_clusters(new_fit, new_cluster_set, mapping, n_samples)

        sse, compactness, dbi = self.post_fit_update(new_centroids, new_labels, new_fuzzy_labels, new_cluster_set,
                                                     data, 0, sample_weights)

        # get counts of each cluster
        # cluster_set.counts = k_n_samples
        # cluster_set.fuzzy_counts = np.sum(k_fit.fuzzy_labels_, axis=0)

        return new_cluster_set, running_time, new_fit.n_iter_, sse, compactness, dbi

    def fit(self, data, n_clusters, sample_weights=None, init=None):
        """
        fit the given data and weights for the given number of clusters, either by kmeans++ or with given initialisation
        centroids
        :param data: N * num_features ndarray of data points to cluster
        :param weights: N * 1 ndarray of weights corresponding to each data point for clustering
        :param optimal_num_clusters: number of clusters to fit for
        :param init: either "kmeans++" for such an initialisation, or a K * num_features ndarray of centroids to be
                     passed as initialisation
        :return: clustering fit object containing centroids, fuzzy_labels, hard labels, inertia (sse),
                 number of iterations
        """

        # convergence is initialised with kmeans++
        # update the clustering object with the given number of clusters with kmeans++ algorithm
        # given the total dataset
        self.clustering_class_args.update([('n_clusters', n_clusters)])
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)
        return self.clustering_obj.fit(data, sample_weights=sample_weights)


def calc_synthetic_inertia(clustering_obj, n_samples, n_features, n_clusters, centers, std_dev):
    """
    function at top-level of module used by multiprocessing Pool to generate synthetic datasets based on
    centroids and standard deviations of the actual dataset, re-cluster them by a given n_clustesr,
    and calculate the total sse (inertia) of many
    bootstrapped datasetss in calc_opt_num_clusters method
    :param clustering_obj: ClusteringAlgorithm object
    :param n_samples: number of data points generated by synthetic datasets
    :param n_features: number of features (dimensions) associated with data points
    :param centers: centroids of actual dataset used to generate synthetic datasets
    :param std_dev: standard deviation of actual dataset used to generate synthetic datasets
    :param n_clusters: number of clusters to fit synthetic datasets to
    :param weights: weights to assign to each of the generated samples for the clustering fit
    :return:
    """
    return clustering_obj.calc_synthetic_inertia(n_samples, n_features, n_clusters, centers, std_dev)


def initialise_clusters(clustering_object, kwargs):
    """
    function at top-level of module used by multiprocessing Pool to re-initialise clusters when
    optimal number of clustesr has changed
    :param clustering_object: ClusteringAlgorithm object
    :param kwargs: keyword arguments passed to initialise_clusters method
    :return:
    """
    return clustering_object.initialise_clusters(**kwargs)


def fit_clusters(clustering_object, kwargs):
    """
    unction at top-level of module used by multiprocessing Pool to re-fit clusters when new data points have come in
    and there is no need to perform a re-initialisation yet
    :param clustering_object: ClusteringAlgorithm object
    :param kwargs: keyword arguments passed to fit_clusters method
    :return:
    """
    return clustering_object.fit_clusters(**kwargs)
