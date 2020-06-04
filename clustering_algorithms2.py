import numpy as np
from numpy.linalg import norm
from datetime import datetime
from FuzzyKMeansPlus import FuzzyKMeansPlus
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs
from copy import deepcopy

"""
NOTES
Random initialisation within know limits in parallel to check for better results
or Planting seeds.at known limits of features to influence centre of masses

To buffer data for re-init or not

COMs are pulled toward new data points by a gravity that is proportional to membership probability/(mass * r^2)

Evolutionary metrics such as mass, inertia and count are probabilistic, carried over by new initialisations and account for 
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


# TODO
#  add averagerunningtime per data sample to csv results
#  Run illustrative example on 20 records, then case study on 3 customers
#  Write 5+ pages on Forgetting effect of data points from over a year ago
#  Clearly define contribution which is applicable to other areas
#  Simplify flowchart
#  max 20 min presesntation with illustrative figures, less equations
#


class ClusterSet:
    """
    Class defining a set of clusters and their attributes for a given problem
    """

    def __init__(self, clusters):
        """
        :param clusters: list of Cluster objects which make up this ClusterSet
        """

        self.num_clusters = len(clusters)

        # lists of most up-to-date attributes clusters. Each list is of length equal to the number of clusters
        self.clusters = clusters

        if len(clusters):
            self.centroids = np.array([cluster.centroid for cluster in clusters])
            self.st_centroids = np.array([cluster.st_centroid for cluster in clusters])
            self.masses = np.array([cluster.mass for cluster in clusters])
            self.errors_absolute = np.array([cluster.error_absolute for cluster in clusters])
            self.errors_squared = np.array([cluster.error_squared for cluster in clusters])

            self.variances = np.array([cluster.variance for cluster in clusters])
            self.cps = np.array([cluster.cp for cluster in clusters])
            self.dbis = np.array([cluster.dbi for cluster in clusters])

            self.inertia = np.sum(self.errors_squared)
            self.cp = np.mean(self.cps)
            self.dbi = np.array([np.mean(self.dbis)])
        else:
            self.centroids = np.zeros(shape=(0, 1))
            self.st_centroids = np.zeros(shape=(0, 1))
            self.masses = np.zeros(shape=(0))
            self.errors_absolute = np.zeros(shape=(0, 1))
            self.errors_squared = np.zeros(shape=(0, 1))

            self.variances = np.zeros(shape=(0, 1))
            self.cps = np.zeros(shape=(0))
            self.dbis = np.zeros(shape=(0))

            self.inertia = 0
            self.cp = 0
            self.dbi = 0

    def update(self, window_size):
        """
        Update the ClusterSet object each time clusters are re-fitted in response to each new data point,
        when fit_clusters is called
        :param fuzzy_labels: 1 * K (number of clusters) array giving the probability that the most recent
                                      incoming data point belongs to each cluster
        """

        # update all cluster attributes
        self.centroids = np.array([cluster.centroid for cluster in self.clusters])
        self.st_centroids = np.array([cluster.st_centroid for cluster in self.clusters])
        self.masses = np.array([cluster.mass for cluster in self.clusters])
        self.variances = np.array([cluster.variance for cluster in self.clusters])
        self.cps = np.array([cluster.cp for cluster in self.clusters])

        # update the number of clusters
        self.num_clusters = len(self.clusters)

        # update all cluster dbi values, if more than one cluster exists
        if self.num_clusters > 1:
            for k in range(self.num_clusters):
                self.clusters[k].dbi = np.max([(self.cps[k] + self.cps[kk])
                              / norm(self.centroids[k] - self.centroids[kk], 2)
                              for kk in range(self.num_clusters) if kk != k])

        # add a new total error_squared, mean compactness or mean dbi value to the aggregate window lists, corresponding
        # to this re-fit
        self.errors_absolute = np.array([cluster.error_absolute for cluster in self.clusters])
        self.errors_squared = np.array([cluster.error_squared for cluster in self.clusters])
        self.dbis = np.array([cluster.dbi for cluster in self.clusters])

        self.inertia = np.sum(self.errors_squared)
        self.cp = np.mean(self.cps)
        self.dbi = np.append(self.dbi, np.mean(self.dbis))

        if self.dbi.shape[0] > window_size + 2:
            self.dbi = np.delete(self.dbi, np.arange(0, window_size + 2 - self.dbi.shape[0] - 1))

    def kill_clusters(self, indices):
        """
        Remove the clusters at the given indices and all attributes associated with them.
        :param indices: a list of indices, at which clusters should be removed from this cluster set
        """

        self.clusters = np.delete(self.clusters, indices, axis=0)
        self.centroids = np.delete(self.centroids, indices, axis=0)
        self.st_centroids = np.delete(self.st_centroids, indices, axis=0)
        self.masses = np.delete(self.masses, indices, axis=0)
        self.variances = np.delete(self.variances, indices, axis=0)
        self.errors_absolute = np.delete(self.errors_absolute, indices, axis=0)
        self.errors_squared = np.delete(self.errors_squared, indices, axis=0)
        self.cps = np.delete(self.cps, indices, axis=0)
        self.dbis = np.delete(self.dbis, indices, axis=0)

        if len(self.errors_squared):
            self.inertia = np.sum(self.errors_squared)
            self.cp = np.mean(self.cps)
            self.dbi = np.mean(self.dbis)

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
            self.st_centroids = np.vstack([self.st_centroids, cluster.st_centroid])
            self.masses = np.append(self.masses, cluster.mass)
            self.variances = np.vstack([self.variances, cluster.variance])
            self.errors_absolute = np.vstack([self.errors_absolute, cluster.error_absolute])
            self.errors_squared = np.vstack([self.errors_squared, cluster.error_squared])
            self.cps = np.append(self.cps, cluster.cp)
            self.dbis = np.append(self.dbis, cluster.dbi)

        # increment number of clusters
        self.num_clusters += len(new_clusters)

        self.inertia = np.sum(self.errors_squared)
        self.cp = np.mean(self.cps)
        self.dbi = np.mean(self.dbis)


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
        # pulled towards new data points by the cluster method update_st_centroid
        self.st_centroid = np.nan

        # number of features, or dimension the data points represented by this cluster
        self.num_features = len(centroid)

        # covariance between different features of the data points represented by this cluster
        self.variance = np.zeros(self.num_features)
        self.error_absolute = np.zeros(self.num_features)
        self.error_squared = np.zeros(self.num_features)
        self.cp = 0.0
        self.dbi = 0.0

        # the mass of a cluster is based on its probabilistic count.
        # Its value decays exponentially with each re-fit in which no incoming data point is closest to it
        # (ie highest membership probability)
        self.mass = 0.0

    def update_errors(self, tol, time_decay_const, errors, fuzzy_weights):
        """
        Update the metrics of this cluster with each incoming data point.
        :param diff: error for each dimension between most recent data point and centroid from this re-fit
        :param fuzzy_weight:
        :return:
        """

        self.error_squared += np.sum((errors ** 2) * fuzzy_weights[:, np.newaxis], axis=0)
        self.error_squared *= tol ** (1 / time_decay_const)

        self.error_absolute += np.sum(errors * fuzzy_weights[:, np.newaxis], axis=0)
        self.error_absolute *= tol ** (1 / time_decay_const)

        self.mass += np.sum(fuzzy_weights, axis=0)
        self.mass *= tol ** (1 / time_decay_const)

    def update_means(self):
        """
        Update the count, probabilistic count, covariance, iteration since last update
        of this cluster with each incoming data point.
        """

        # if a batch of new data points have been added to the cluster, update the compactness and variance
        # and time since last update

        # update compactness
        if self.mass:
            self.cp = np.sum(self.error_absolute) / self.mass

        # update variance
        if self.mass > 1.0:
            self.variance = self.error_squared / (self.mass - 1)

    def update_st_centroid(self, data, fuzzy_weights):
        pass


class GravitationalCluster(Cluster):
    """
    Clustering object with a) centre-of-mass attribute which is influenced by the gravitational pull of new data points,
    generally follow trends in the data that are not captured by the centroid and are used to decide which clusters
    should be merged or split and b) mass attribute which decays with time as the cluster is not updated and is used
    to kill significantly lightweight clusters and to move lightweight clusters towards new data trends in order to 
    merge them or split them if necessary.
    """

    def __init__(self, centroid, sample_count_initialised=0, time_decay_const=48):
        super().__init__(centroid=centroid, sample_count_initialised=sample_count_initialised)
        # proportionality constant used when pulling centre-of-masses towards new data points
        self.time_decay_const = time_decay_const
        # centre-of-mass of the cluster, updated with each new data point added
        self.st_centroid = centroid.copy()
        self.st_mass = 0.0

    def update_st_centroid(self, data, fuzzy_weights):
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

        fuzzy_weights_sum = np.sum(fuzzy_weights)
        self.st_mass += fuzzy_weights_sum
        if self.st_mass:
            self.st_centroid += (np.sum((data * fuzzy_weights[:, np.newaxis])
                                           - (self.st_centroid * fuzzy_weights_sum), axis=0)) / \
                                   self.st_mass



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

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size=None, fuzziness=1,
                 max_iter=300, tol=0.0001, alpha=None, time_decay_const=None):

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

        # number counts after which a cluster should be killed
        self.time_decay_const = time_decay_const

        # weights for each point in data_buf
        self.sample_weights = np.array([])

        self.window_size = 0

    def calc_synthetic_inertia(self, n_samples, n_features, n_clusters, centers, std_dev):
        """
        given the centroids and standard deviation of a representative dataset, generate a
        bootstrapped datasets, cluster them to the given n_clusters and get their inertia) values
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

    def sort_clusters(self, new_fit, mapping, n_samples):
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

        return new_centroids, new_labels, new_fuzzy_labels

    def initialise_clusters(self, data, cluster_set, optimal_num_clusters, sample_count, weighted_centroids,
                            mass_normaliser, **kwargs):
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
        :return error_squared: approximate total error_squared
        """

        data_points, sample_weights = self.generate_fit_input_data(data, cluster_set, weighted_centroids)

        # fit the centroids and buffered data points with kmeans++ initialisation
        running_time = datetime.now()
        init_fit = self.fit(data_points, optimal_num_clusters, weights=sample_weights, init="kmeans++")

        # get the run time, number of iterations to convergence and centroids
        running_time = (datetime.now() - running_time).total_seconds()
        num_iters = init_fit.n_iter_
        new_centroids = init_fit.cluster_centers_

        mapping = self.map_clusters(cluster_set.num_clusters, optimal_num_clusters, cluster_set.centroids,
                                    new_centroids)

        # re-arrange new clusters and fit results to align with old
        new_centroids, new_labels, new_fuzzy_labels = \
            self.sort_clusters(init_fit, mapping, len(data_points))

        new_clusters = np.array([self.cluster_class(new_centroids[k], sample_count_initialised=sample_count, **kwargs)
                                 for k in range(len(new_centroids))])
        # reinitialise new_cluster_set based on unsorted list of clusters
        new_cluster_set = ClusterSet(new_clusters)

        # if old clusters exist
        # carry over normalised counts, probabilist_counts and masses
        # such that historic masses of clusters are not disregarded but also do not dominate
        # the unit mass of new data points
        if cluster_set.num_clusters:

            # mass_normaliser = np.min(cluster_set.masses[cluster_set.masses > 0])
            #np.min(cluster_set.masses[cluster_set.masses > 0])
            #norm(cluster_set.masses, 2)

            abs_error_normaliser = mass_normaliser
            #l1_error = np.sum(cluster_set.errors_absolute, axis=1)
            # zero_mask = np.ma.masked_equal(cluster_set.errors_absolute, 1)
            # abs_error_normaliser = np.min(zero_mask, axis=0) if len(zero_mask) else 1.0
            #np.min(cluster_set.errors_absolute[cluster_set.errors_absolute > 0], axis=0)
            #norm(cluster_set.errors_absolute, 2, axis=0)
            #np.max(l1_error[l1_error > 1]) if len(l1_error[l1_error > 1]) else 1

            sq_error_normaliser = mass_normaliser
            # l2_error = np.sum(cluster_set.errors_squared, axis=1)**0.5
            # zero_mask = np.ma.masked_equal(cluster_set.errors_squared, 1)
            # sq_error_normaliser = np.min(zero_mask, axis=0) if len(zero_mask) else 1.0
            #np.min(cluster_set.errors_squared[cluster_set.errors_squared > 0], axis=0)
            # #norm(cluster_set.errors_squared, 2, axis=0)
            #np.max(l2_error[l2_error > 1]) if len(l2_error[l2_error > 1]) else 1

            # loop through the new clusters and initialise their counts and masses based on the pre-existing
            # clusters if they exist
            for k in range(np.min([cluster_set.num_clusters, new_cluster_set.num_clusters])):
                new_cluster_set.clusters[k].error_absolute = cluster_set.errors_absolute[k] / abs_error_normaliser
                new_cluster_set.clusters[k].error_squared = cluster_set.errors_squared[k] / sq_error_normaliser
                new_cluster_set.clusters[k].mass = cluster_set.masses[k] / mass_normaliser

        inertia, compactness, dbi = self.post_fit_update(new_centroids, new_fuzzy_labels, new_cluster_set,
                                                         data_points, sample_weights, False)

        return new_cluster_set, running_time, num_iters, inertia, compactness, dbi

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
        :return error_squared: approximate total error_squared
        """

        # get data points and weights
        data_points, sample_weights = self.generate_fit_input_data(data, cluster_set, weighted_centroids)

        # fit the centroids and buffered data points passing the centroids as initialisation instead of using kmeans++
        running_time = datetime.now()

        fit = self.fit(data_points, optimal_num_clusters, weights=sample_weights, init=cluster_set.centroids)

        # get the run time, number of iterations to convergence and centroids
        running_time = (datetime.now() - running_time).total_seconds()
        num_iters = self.clustering_obj.n_iter_
        new_centroids = fit.cluster_centers_

        # sorting index for new_centroids
        mapping = self.map_clusters(cluster_set.num_clusters, optimal_num_clusters, cluster_set.centroids,
                                    new_centroids)

        new_centroids, new_labels, new_fuzzy_labels = self.sort_clusters(fit, mapping, len(data))

        inertia, compactness, dbi = self.post_fit_update(new_centroids, new_fuzzy_labels,
                                                         cluster_set, data, sample_weights[0:len(data)], True)

        return cluster_set, running_time, num_iters, inertia, compactness, dbi

    def post_fit_update(self, new_centroids, new_fuzzy_labels, cluster_set, data, sample_weights, fit_clusters):
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
        #for d in range(len(data)):
        # loop through all existing clusters
        for k in range(len(new_centroids)):

            # update error_absolute, error_squared and mass weighted by fuzzy weight
            cluster_set.clusters[k].update_errors(tol=self.tol, time_decay_const=self.time_decay_const,
                                                  errors=np.abs(data - new_centroids[k]),
                                                  fuzzy_weights=new_fuzzy_weights[:, k])

            # update compactness and variance
            cluster_set.clusters[k].update_means()

            # update total fuzzy weight added since last init to track short term trends
            fuzzy_weights_sum = np.sum(new_fuzzy_weights[:, k])
            cluster_set.clusters[k].st_mass += fuzzy_weights_sum

            # update long-term centroid and short term centroid
            if fit_clusters:

                # updated based on existing centroids and incoming data points
                cluster_set.clusters[k].centroid = new_centroids[k]

                # short term centroids only updated based on incoming data points
                if cluster_set.clusters[k].st_mass:
                    cluster_set.clusters[k].st_centroid += \
                        np.sum((data * new_fuzzy_weights[:, k])
                                - (cluster_set.clusters[k].st_centroid * fuzzy_weights_sum), axis=0) \
                        / cluster_set.clusters[k].st_mass


        # update ClusterSet metrics and DBI, compactness and variance weighted by fuzzy weight
        cluster_set.update(window_size=self.window_size)

        return cluster_set.inertia, cluster_set.cp, cluster_set.dbi[-1]

    def feed_data(self, new_data, new_sample_weights=None):
        """
        update self.data_buf with clean new incoming data points
        :param new_data: pandas row of data
        """
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

        self.time_decay_const = np.inf

    def fit(self, data, n_clusters, weights=None, init="kmeans++"):
        """
        fit the given data and weights for the given number of clusters, either by kmeans++ or with given initialisation
        centroids
        :param data: N * num_features ndarray of data points to cluster
        :param weights: N * 1 ndarray of weights corresponding to each data point for clustering
        :param n_clusters: number of clusters to fit for
        :param init: either "kmeans++" for such an initialisation, or a K * num_features ndarray of centroids to be
                     passed as initialisation
        :return: clustering fit object containing centroids, fuzzy_labels, inertia), number of iterations
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
        :return error_squared: approximate online error_squared
        """

        # declare running time, number of iterations and error_squared for this convergence
        running_time = np.nan
        num_iters = np.nan
        inertia = np.nan
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

        # get the smallest nonzero mass
        mass_normaliser = np.min(self.cluster_set.masses[self.cluster_set.masses > 1.0]) \
            if len(self.cluster_set.masses[self.cluster_set.masses > 1.0]) else 1.0

        # re-initialise clusters if necessary
        if re_init:
            self.cluster_set, running_time, num_iters, inertia, compactness, dbi = \
                self.initialise_clusters(self.data_buf, self.cluster_set, self.optimal_num_clusters,
                                         mass_normaliser=mass_normaliser, sample_count=sample_count,
                                         weighted_centroids=False)
            self.n_samples_since_last_init = len(self.data_buf)

            # flush old data from memory
            self.flush_data()

        # otherwise re-fit clusters if possible
        elif re_fit:

            self.cluster_set, running_time, num_iters, inertia, compactness, dbi = \
                self.fit_clusters(self.data_buf, self.cluster_set, self.optimal_num_clusters,
                                  sample_count=sample_count, weighted_centroids=False)
            self.n_samples_since_last_init += len(self.data_buf)

            # flush old data from memory
            self.flush_data()

        return self.cluster_set, running_time, num_iters, inertia, compactness, dbi


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

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size,
                 time_decay_const=100, fuzziness=1, alpha=0.1, max_iter=10000, tol=0.00006, window_size=2):
        super().__init__(features=features, init_num_clusters=init_num_clusters, batch_size=batch_size,
                         init_batch_size=init_batch_size, fuzziness=fuzziness, max_iter=max_iter, tol=tol,
                         alpha=alpha, time_decay_const=time_decay_const)

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

        # window of convergences over which to check for continuously increasing DBI, indicating that clusters should
        # re-initialised
        self.window_size = window_size

    def calc_opt_num_clusters(self, pool):
        """
        given the cluster parameters of datasets clustered into K-1, K and K+1 clusters,
        calculate the optimal number of clusters by generating bootstrap synthetic data sets, calculating
        the pdf of their error_squared values and calculating the cd of these pdf functions up to the error_squared achieved
        for the next greatest number of clusters
        :param pool: multiprocessing pool object
        :return new_optimal_num_clusters most optimal number of clusters for this dataset,
        either equal to original or +/- 1
        """

        # number of features, samples to use for synthetic datasets
        n_features = len(self.features)

        # assume first that optimal number of clusters is current value +1, if not it will be reset in the following loop
        new_optimal_num_clusters = self.optimal_num_clusters[-1]

        # number of synthetic datasets to generate
        num_syn_datasets = 10
        # number of bins to form error_squared histogram
        num_bins = 100

        # loop through the already clustered datasets for K-1 and K, cluster synthetic datasets based on these values,
        # compare the resulting synthetic error_squared values to the error_squared value for the already clustered K+1 cluster solution
        for c in range(2):

            # K, number of clusters with which to cluster synthetic datasets
            k_n_clusters = self.optimal_num_clusters[c]

            # if there only exists one cluster here, then this is not a realistic value to reset K to, skip it
            if k_n_clusters == 1:
                continue

            # get the already calculated centroids and standard deviation for clustering at K number of clusters
            k_centroids = self.cluster_set[c].centroids

            # square root of variance accrued since last initialisation
            k_std_dev = np.sqrt(self.cluster_set[c].variances)

            # data samples assigned to each clusters
            n_samples_per_cluster = np.asarray(np.ceil(self.cluster_set[c].masses), dtype='int')

            synthetic_inertias = pool.starmap(calc_synthetic_inertia,
                                              [(self, n_samples_per_cluster, n_features, k_n_clusters, k_centroids,
                                                k_std_dev)
                                               for s in range(num_syn_datasets)])

            # get the error_squared value of already clustered dataset for K+1 number clusters
            # as this number reflects only the addition of data points to clusters as they arrive, and not how
            # these data points would later be reassigned to reduce error_squared, this value is overestimated, thus the median
            # is used to reduce it to a more reasonable value

            # error_squared accrued by K+1 algorithm run
            # todo this should decrease for greater number of clusters
            #  it is increasing because it is normalised by a smaller number for c+1 than for c
            kplus_inertia = self.cluster_set[c + 1].inertia

            # calculate pdf of error_squared values achieved over all of these synthetic data sets
            pdf, bin_edges = np.histogram(synthetic_inertias, range=(0, np.max(synthetic_inertias)),
                                          bins=num_bins, density=True)

            # pdf must be integrated over interval to yield pmf
            pmf = pdf * np.diff(bin_edges)

            # calculate cdf
            cdf = np.cumsum(pmf)

            # get p-value = cdf of error_squared pdf at the error_squared achieved for the next greatest number of clusters K+1
            # p value = 1 means that it is certain that the value of the error_squared for K
            #  (from bootstrapped datasets) is less than the actual error_squared for K+1, so we should stop at K

            # p value = 0 means that it is certain that the value of the error_squared for K
            #   (from bootstrapped datasets) is greater than the actual error_squared for K+1, so we should increase to K+1

            p_value = np.interp(kplus_inertia, bin_edges[0:-1], cdf)

            # if there is a sufficiently high probability (approx 5-10%)
            # that the error_squared resulting from clustering at K based on synthetic datasets
            # is less than error_squared resulting from clustering at K+1, then update the optimal number of clusters to K and
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
        a = 1 - (live_centroid_dead_centroid_distances / np.max(live_centroid_dead_centroid_distances)) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            a /= np.sum(a)
        a[np.isnan(a)] = 1

        # kill the outlier cluster
        new_cluster_set.kill_clusters([outlier_idx])

        print(f"Outlier cluster killed.")

        # update original clusters
        for k in range(new_cluster_set.num_clusters):
            new_cluster_set.clusters[k].error_absolute += a[k] * cluster_set.errors_absolute[outlier_idx]
            new_cluster_set.errors_absolute[k] = new_cluster_set.clusters[k].error_absolute

            new_cluster_set.clusters[k].error_squared += a[k] * cluster_set.errors_squared[outlier_idx]
            new_cluster_set.errors_squared[k] = new_cluster_set.clusters[k].error_squared

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
                                             time_decay_const=self.time_decay_const)

        # calculate euclidean distances between all centroids and the outlier centroid
        old_centroid_new_centroid_distances = euclidean_distances(cluster_set.centroids,
                                                                  self.data_buf[outlier_idx, np.newaxis])

        # generate coefficients between 0 and 1 which indicate what proportion of the metrics
        # (count, mass, fuzzy_count) of each old cluster will be passes on to this new cluste
        a = ((old_centroid_new_centroid_distances / np.max(old_centroid_new_centroid_distances)) ** 2).squeeze(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            a /= np.sum(a)
        a[np.isnan(a)] = 1

        outlier_cluster.mass = np.sum((1 - a) * cluster_set.masses)
        outlier_cluster.error_absolute = np.sum((1 - a)[:, np.newaxis] * cluster_set.errors_absolute, axis=0)
        outlier_cluster.error_squared = np.sum((1 - a)[:, np.newaxis] * cluster_set.errors_squared, axis=0)

        new_cluster_set.create_clusters([outlier_cluster])

        print(f"Outlier cluster created.")

        # update original clusters
        for k in range(cluster_set.num_clusters):
            new_cluster_set.clusters[k].mass = a[k] * cluster_set.masses[k]
            new_cluster_set.masses[k] = new_cluster_set.clusters[k].mass

            new_cluster_set.clusters[k].error_absolute = a[k] * cluster_set.errors_absolute[k]
            new_cluster_set.errors_absolute[k] = new_cluster_set.clusters[k].error_absolute

            new_cluster_set.clusters[k].error_squared = a[k] * cluster_set.errors_squared[k]
            new_cluster_set.errors_squared[k] = new_cluster_set.clusters[k].error_squared

        return new_cluster_set

    def split_cluster(self, cluster_set):
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

        # if self.gravitational_const:
        # calculate euclidean distances between all centroids and st_centroids
        centroid_com_distances = \
            paired_distances(cluster_set.centroids, cluster_set.st_centroids)

        # get index of cluster with largest centroid - centre-of-mass difference
        split_idx = np.argmax(centroid_com_distances)

        # set new centroids to centroid and centre-of-mass of cluster to split
        new_centroids = np.vstack([cluster_set.centroids[split_idx], cluster_set.st_centroids[split_idx]])
        # else:
        # get index of cluster with largest variance
        # cluster_std_devs = np.sum(cluster_set.variances, axis=1) ** 0.5
        # split_idx = np.argmax(cluster_std_devs, axis=0)

        # get index of new data point closest to it
        # point_centroid_distances = euclidean_distances(self.data_buf, cluster_set.centroids[split_idx, np.newaxis])

        # get index of minimum distance between new data points and their closest centroids
        # close_point_idx = np.argmin(point_centroid_distances, axis=0)

        # set new centroids to centroid and centroid + std_dev of cluster to split
        # new_centroids = np.vstack([cluster_set.centroids[split_idx], self.data_buf[close_point_idx]])

        print(f"Cluster {split_idx} split.")

        # kill the old clusters
        new_cluster_set.kill_clusters([split_idx])

        # create the new clusters
        new_cluster_set.create_clusters([self.cluster_class(centroid=new_centroids[idx],
                                                            time_decay_const=self.time_decay_const)
                                         for idx in range(len(new_centroids))])

        # half the counts, fuzzy_counts and masses and add to the new split clusters
        for k in [-2, -1]:
            new_cluster_set.clusters[k].error_absolute += cluster_set.errors_absolute[split_idx] / 2
            new_cluster_set.errors_absolute[k] = new_cluster_set.clusters[k].error_absolute

            new_cluster_set.clusters[k].error_squared += cluster_set.errors_squared[split_idx] / 2
            new_cluster_set.errors_squared[k] = new_cluster_set.clusters[k].error_squared

            new_cluster_set.clusters[k].mass += cluster_set.masses[split_idx] / 2
            new_cluster_set.masses[k] = new_cluster_set.clusters[k].mass

        return new_cluster_set

    def merge_clusters(self, cluster_set, merge_indices, sample_count):

        # copy original cluster_set
        new_cluster_set = deepcopy(cluster_set)

        # the new centroid is the weighted mean of the merging clusters
        new_centroid = np.sum(cluster_set.masses[merge_indices, np.newaxis] * cluster_set.centroids[merge_indices, :],
                              axis=0) / np.sum(cluster_set.masses[merge_indices])

        # kill the original clusters
        new_cluster_set.kill_clusters(merge_indices)

        # create the new cluster
        new_cluster_set.create_clusters([
            self.cluster_class(centroid=new_centroid,
                               time_decay_const=self.time_decay_const,
                               sample_count_initialised=sample_count)])

        # update the online variance, inertia and mass the new merged cluster

        new_cluster_set.clusters[-1].error_absolute = np.sum(cluster_set.errors_absolute[merge_indices], axis=0)
        new_cluster_set.errors_absolute[-1] = new_cluster_set.clusters[-1].error_absolute

        new_cluster_set.clusters[-1].error_squared = np.sum(cluster_set.errors_squared[merge_indices], axis=0)
        new_cluster_set.errors_squared[-1] = new_cluster_set.clusters[-1].error_squared

        new_cluster_set.clusters[-1].mass = np.sum(cluster_set.masses[merge_indices])
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

        # calculate distances separating centroids and isolate unique nonzero differences
        tril_indices = np.tril_indices(cluster_set.num_clusters, k=-1)
        centroid_distances = pairwise_distances(cluster_set.centroids)[tril_indices]

        # get zscore of distances separating centroids
        # centroid_z = stats.zscore(centroid_distances, axis=None)
        # centroid_z_outlier_index = np.argmin(centroid_z)
        centroid_outlier_index = np.argmin(centroid_distances)

        # get index of clusters with lowest centroid separation
        centroid_outlier_index = np.array([tril_indices[0][centroid_outlier_index],
                                           tril_indices[1][centroid_outlier_index]])

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
        :return error_squared: approximate error_squared
        """

        # declare running time, number of iterations and error_squared variables
        running_time = np.empty(3)
        running_time[:] = np.nan
        num_iters = np.empty(3)
        num_iters[:] = np.nan
        error_squared = np.empty(3)
        error_squared[:] = np.nan
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

        elif self.n_samples_since_last_init > self.window_size + diff_order \
                and np.all(np.diff(self.cluster_set[opt_index].dbi, n=diff_order)[-self.window_size:] > 0):

            # if self.n_samples_since_last_init >= self.init_batch_size:
            #     print(f"Number of samples since the last initialisation has exceeded initial batch size parameter"
            #           f" {self.init_batch_size}.")
            # else:
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
                          f"Expand cluster by splitting, then re-initialise clusters.")
                    for c in range(3):
                        self.cluster_set[c] = self.split_cluster(cluster_set=self.cluster_set[c])

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
                          f"Reduce clusters by merging, then re-initialise clusters.")
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
            # inter_centroid_distances = inter_centroid_distances[np.where(inter_centroid_distances != 0)]

            # get index of maximum distance between any two centroids
            max_inter_centroid_distance = np.max(inter_centroid_distances)

            # if the distance from the most outlying data point to its closest centroid is greater than any
            # inter-centroidal distance, then create a new cluster from the outlier
            if min_point_centroid_distance[outlier_idx] > max_inter_centroid_distance:

                # increment K
                self.optimal_num_clusters = [n + 1 for n in self.optimal_num_clusters]

                # create new clusters in each of the three parallel algorithm runs for K-1, K and K+1
                for c in range(3):
                    self.cluster_set[c] = self.create_outlier_cluster(self.cluster_set[c], outlier_idx)

                re_init = True

            # get index of cluster with neglibible mass
            negligible_mass_indices = np.argwhere((self.cluster_set[opt_index].masses /
                                                  np.max(self.cluster_set[opt_index].masses))**2 < self.tol)
            # if the smallest cluster mass is less than the allowed tolerance
            if len(negligible_mass_indices):
                for negligible_mass_idx in negligible_mass_indices[0]:
                    if self.optimal_num_clusters[0] >= 2:
                        # decrement K
                        self.optimal_num_clusters = [n - 1 for n in self.optimal_num_clusters]

                        # kill or merge new clusters in each of the three parallel algorithm runs for K-1 and K+1
                        for c in range(3):
                            negligible_mass_idx = np.argmin(self.cluster_set[c].masses)
                            self.cluster_set[c] = self.kill_outlier_cluster(self.cluster_set[c], negligible_mass_idx)

                        re_init = True

            # get indices of centroids which have converged on eachother
            # get index of 0 distance between any two centroids
            inter_centroid_distances = euclidean_distances(self.cluster_set[opt_index].centroids)
            tril_indices = np.tril_indices(self.cluster_set[opt_index].num_clusters, k=-1)
            converged_indices = np.argwhere((inter_centroid_distances /
                                            np.max(inter_centroid_distances))**2 < self.tol)
            converged_indices = [idx for idx in converged_indices
                                 if idx[0] in tril_indices[0] and idx[1] in tril_indices[1] and idx[0] != idx[1]]
            # converged_indices = np.unique(converged_indices)
            # if the smallest cluster mass is less than the allowed tolerance
            for converged_idx in converged_indices:
                if self.optimal_num_clusters[0] >= 2:
                    # decrement K
                    self.optimal_num_clusters = [n - 1 for n in self.optimal_num_clusters]

                    # kill or merge new clusters in each of the three parallel algorithm runs for K-1, K and K+1
                    for c in range(3):
                        self.cluster_set[c] = self.reduce_clusters(self.cluster_set[c], sample_count)

                    re_init = True

            if not re_init:
                print(f"No outliers have been found and there is no need to re-initialise clusters yet."
                      f" Re-fit clusters.")
                re_fit = True

        # re-initialise clusters if necessary for each of the algorithm runs
        if re_init:

            # get the smallest nonzero mass over all cluster sets
            all_cluster_masses = np.concatenate([cs.masses for cs in self.cluster_set])
            mass_normaliser = np.min(all_cluster_masses[all_cluster_masses > 1.0]) \
                if len(all_cluster_masses[all_cluster_masses > 1.0]) else 1.0

            fit_results = pool.starmap(initialise_clusters,
                                       [(self, {'data': self.data_buf, 'cluster_set': self.cluster_set[c],
                                                'optimal_num_clusters': self.optimal_num_clusters[c],
                                                'sample_count': sample_count,
                                                'weighted_centroids': True,
                                                'mass_normaliser': mass_normaliser,
                                                'time_decay_const': self.time_decay_const}) for c in
                                        range(3)])

            self.n_samples_since_last_init = len(self.data_buf)

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
            error_squared = [res[3] for res in fit_results]
            compactness = [res[4] for res in fit_results]
            dbi = [res[5] for res in fit_results]

            # flush old data from memory if it was used to fit clustering
            self.flush_data()

        return self.cluster_set[opt_index], running_time[opt_index], num_iters[opt_index], error_squared[opt_index], \
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
        :return: clustering fit object containing centroids, fuzzy_labels, hard labels, inertia),
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


class OfflineOptimalKMeansPlus(ClusteringAlgorithm):
    """
    Class defining Offline OptimalKMeansPlus ClusteringAlgorithm as tuning protocol and baseline to online version.
    """

    def __init__(self, features, init_num_clusters, batch_size, init_batch_size, fuzziness=2,
                 max_iter=10000, tol=0.000006, alpha=None, time_decay_const=None,
                 cluster_set=None):

        super().__init__(features=features, init_num_clusters=init_num_clusters, batch_size=batch_size,
                         init_batch_size=init_batch_size, fuzziness=fuzziness, max_iter=max_iter, tol=tol, alpha=alpha,
                         time_decay_const=time_decay_const)

        # see notes from parent class ClusteringAlgorithm

        self.clustering_class = FuzzyKMeansPlus
        self.clustering_class_args = {'init': 'kmeans++', 'fuzziness': self.fuzziness, 'max_iter': self.max_iter,
                                      'tol': self.tol}
        self.clustering_obj = self.clustering_class(**self.clustering_class_args)

        self.cluster_class = GravitationalCluster

        self.optimal_num_clusters = init_num_clusters

        self.cluster_set = ClusterSet([]) if cluster_set is None else cluster_set

    def update_clusters(self, pool):

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
        :return error_squared: total inertia) since last initialisation
        """

        n_samples, n_features = data.shape

        # number of synthetic datasets to generate
        num_syn_datasets = 100

        # number of bins to form error_squared histogram
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

            for k in range(k_n_clusters):
                if k_n_probabilistic_samples[k] > 1:
                    k_std_dev[k] = np.sqrt((1 / (k_n_probabilistic_samples[k] - 1))
                                           * np.sum(k_fuzzy_labels[:, k, np.newaxis] ** self.fuzziness
                                                    * (data - k_centroids[k]) ** 2, axis=0))
            # generate synthetic datasets based on the centroids and standard deviations
            # of actual dataset clustered with K clusters
            # cluster the synthetic datasets with K clusters
            # fetch the resulting error_squared values
            print(f"Calculating synthetic inertia values for n_clusters = {c}.")

            synthetic_inertias = pool.starmap(calc_synthetic_inertia,
                                              [(self, np.asarray(np.ceil(k_n_probabilistic_samples), dtype='int'),
                                                n_features, k_n_clusters, k_centroids, k_std_dev)
                                               for s in range(num_syn_datasets)])

            # fit the actual dataset with K+1 clusters and get the resulting error_squared
            kplus_n_clusters = c + 1
            kplus_fit = self.fit(data, kplus_n_clusters, sample_weights=sample_weights)
            kplus_inertia = kplus_fit.inertia_

            # calculate the pdf of the synthetic_inertia values
            pdf, bin_edges = np.histogram(synthetic_inertias, range=(0, np.max(synthetic_inertias)), bins=num_bins,
                                          density=True)

            pmf = pdf * np.diff(bin_edges)

            # calculate the cdf of the synthetic_inertia values
            cdf = np.cumsum(pmf)

            # calculate the p-value = value of cdf of synthetic_inertia values at the error_squared value of the actual
            # K+1 clustering = probability of finding a synthetic K-clustered dataset with a error_squared lower than
            # the error_squared of the actual K+1-clustered dataset
            p_value = np.interp(kplus_inertia, bin_edges[0:-1], cdf)

            # if the p-value is significant, then we choose K as the optimal number of clusters
            if p_value > self.alpha or c == n_samples - 1:
                new_fit = k_fit
                new_n_clusters = k_n_clusters
                new_centroids = k_centroids
                break

        # get the running time
        running_time = (datetime.now() - running_time).total_seconds()

        # re-arrange new clusters and fit results to align with old
        mapping = self.map_clusters(old_n_clusters, new_n_clusters, old_centroids, new_centroids)
        new_centroids, new_labels, new_fuzzy_labels = self.sort_clusters(new_fit, mapping, n_samples)

        # create the new ClusterSet
        kwargs = {}
        new_clusters = np.array([self.cluster_class(new_centroids[k], sample_count_initialised=0, **kwargs)
                                 for k in range(len(new_centroids))])
        # reinitialise new_cluster_set based on unsorted list of clusters
        new_cluster_set = ClusterSet(new_clusters)

        inertia, compactness, dbi = self.post_fit_update(new_centroids, new_fuzzy_labels, new_cluster_set, data,
                                                         sample_weights, False)

        return new_cluster_set, running_time, new_fit.n_iter_, inertia, compactness, dbi

    def fit(self, data, n_clusters, sample_weights=None, init=None):
        """
        fit the given data and weights for the given number of clusters, either by kmeans++ or with given initialisation
        centroids
        :param data: N * num_features ndarray of data points to cluster
        :param weights: N * 1 ndarray of weights corresponding to each data point for clustering
        :param optimal_num_clusters: number of clusters to fit for
        :param init: either "kmeans++" for such an initialisation, or a K * num_features ndarray of centroids to be
                     passed as initialisation
        :return: clustering fit object containing centroids, fuzzy_labels, hard labels, inertia),
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
    and calculate the total error_squared (inertia) of many
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
