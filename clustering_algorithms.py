import numpy as np
from numpy.linalg import norm
from numpy.random import choice, randint, random
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, paired_distances, cosine_distances
from datetime import datetime


class Cluster:
    def __init__(self, mean, total_count, life, kernel_count):
        self.mean = mean
        self.total_count = total_count
        self.life = life
        self.kernel_count = kernel_count


class Feature:
    def __init__(self, name, lb=None, ub=None, step=None):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.step = step
        self.live = True


class ClusteringAlgorithm:

    def __init__(self, clustering_results, features):
        # N (number of data points) * D (number of dimensions) array of data stream coordinates
        self.data_buf = np.array([])

        self.features = features

        # K * D array of centroid coordinates, variances, data point count
        self.centres = np.array([])
        self.cohesion = np.array([])
        self.separation = np.array([])
        self.count = np.array([])

        self.clustering_results = clustering_results

    def add_feature(self, name, lb=None, ub=None, step=None):
        self.features = np.append(Feature(name=name, lb=lb, ub=ub, step=step))
        # add new column to centres for new dimension and reinitialise this feature space
        self.centres = np.c_(self.centres, np.zeros(len(self.centres)))

    def remove_feature(self):
        pass

    def merge_clusters(self, k_indices):
        """
        merge clusters of the given indices
        if 2 clusters merge, then there exists one less cluster to account for
        """
        # calculate new centroid and count
        new_centroid = np.sum(self.count[k_indices] * self.centres[k_indices]) / len(k_indices)
        new_count = np.sum(self.count[k_indices])

        # remove old cluster data
        self.centres = np.delete(self.centres, k_indices)
        self.cohesion = np.delete(self.cohesion, k_indices)
        self.separation = np.delete(self.separation, k_indices)
        self.count = np.delete(self.count, k_indices)

        # add new cluster data
        self.centres = np.r_(self.centres, new_centroid)
        self.cohesion = np.r_(self.cohesion, 0)
        self.separation = np.r_(self.separation, 0)
        self.count = np.r_(self.count, new_count)

    def split_clusters(self, k_index):
        """
        split the cluster at the given index, and add new to end of cluster data arrays
        if 2 clusters split, then there exists one more cluster to account for
        """
        # calculate new centres and counts
        # TODO some approximated function of old cluster centroid, separation, cohesion and count
        new_centroids = np.ones(2, len(self.features))
        new_counts = np.ones(2, 1)

        # remove old cluster data
        self.centres = np.delete(self.centres, k_index)
        self.cohesion = np.delete(self.cohesion, k_index)
        self.separation = np.delete(self.separation, k_index)
        self.count = np.delete(self.count, k_index)

        # add new cluster data
        self.centres = np.r_(self.centres, new_centroids)
        self.cohesion = np.r_(self.cohesion, 0)
        self.separation = np.r_(self.separation, 0)
        self.count = np.r_(self.count, new_counts)

    def kill_cluster(self):
        pass

    def feed_data(self, new_data):

        if len(self.data_buf) == 0:
            self.data_buf = np.expand_dims(new_data[[feat.name for feat in self.features]].values, axis=0)
        else:
            self.data_buf = np.vstack((self.data_buf, new_data[[feat.name for feat in self.features]].values))

    def flush_data(self):
        self.data_buf = np.array([])


class OnlineK(ClusteringAlgorithm):

    def __init__(self, clustering_results, features, num_clusters, max_iter=10000, tol=0.0001, learning_rate=0.05,
                 distance_type='euclidean', init_type='random'):

        super().__init__(clustering_results, features)
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.distance_type = distance_type
        self.init_type = init_type

        # # 1 * K array of the number of data points in each cluster
        # self.cluster_counts = [0 for k in range(len(num_clusters))]

    def get_distances(self):
        if self.distance_type == 'euclidean':
            return euclidean_distances(self.data_buf, self.centres)
        elif self.distance_type == 'manhattan':
            return manhattan_distances(self.data_buf, self.centres)
        elif self.distance_type == 'cosine':
            return cosine_distances(self.data_buf, self.centres)

    def update_clusters(self):

        if np.isnan(self.data_buf).any():
            self.flush_data()
        else:
            # if centres have not yet been initialised
            if len(self.centres) == 0:
                # if K*K number of data points have arrived, initialise cluster means
                if len(self.data_buf) == self.num_clusters ** 2:
                    self.initialise_clusters()
            # else if cluster centres have already been initialised, feed data to stream to update clusters
            else:
                self.converge()

                # clear old data from memory
                self.flush_data()

    def initialise_clusters(self, full_init=True):

        """
        k-means++ initialisation
        :return:
        """

        # if this is a full initialisation of all feature dimensions TODO
        if full_init:
            self.centres = np.zeros(shape=(self.num_clusters, len(self.features)))
            self.cohesion = np.zeros(shape=(self.num_clusters, 1))
            self.separation = np.zeros(shape=(self.num_clusters, 1))
            self.count = np.zeros(shape=(self.num_clusters, 1))

        if self.init_type == 'random':
            for k in range(len(self.centres)):
                prob_dist = np.ones(len(self.data_buf), 1) / len(self.data_buf)
                i = choice(range(len(self.data_buf)), 1, p=prob_dist)
                self.centres[k, :] = self.data_buf[i]

        elif self.init_type == 'semi-random':

            f_known = [f for f in range(len(self.features)) if
                       self.features[f].live and self.features[f].lb is not None]
            f_unknown = [f for f in range(len(self.features)) if self.features[f].live and self.features[f].lb is None]

            # loop through features of known bounds
            # if this feature has finite and discrete values, initialise these dimensions of the clusters
            # distributed uniformly over these
            for f in f_known:

                for k in range(len(self.centres)):
                    self.centres[k, f] = np.floor((self.features[f].lb +
                                                     ((self.features[f].ub - self.features[f].lb) * random()))
                                                    / self.features[f].step) * self.features[f].step

            # for unknown feature spaces, distribute as with k++
            # randomly initialise first centroid out of available data points
            self.centres[0, f_unknown] = self.data_buf[randint(low=0, high=len(self.data_buf)), f_unknown]

            for k in range(1, len(self.centres)):
                # for each data point compute its distance from the nearest, previously chosen centroid
                distances = euclidean_distances(self.data_buf[:, f_unknown], self.centres[0:k, f_unknown])
                weights = np.min(distances, axis=1)
                prob_dist = weights / np.sum(weights)

                # select the next centroid from the data points such that the probability of choosing a point as centroid is
                # directly proportional to its distance from the nearest, previously chosen centroid

                i = choice(range(len(self.data_buf)), 1, p=prob_dist)
                self.centres[k, f_unknown] = self.data_buf[i, f_unknown]

        elif self.init_type == 'k++':

            # randomly initialise first centroid out of available data points
            self.centres[0] = self.data_buf[randint(low=0, high=len(self.data_buf))]

            for k in range(1, len(self.centres)):
                # for each data point compute its distance from the nearest, previously chosen centroid
                distances = euclidean_distances(self.data_buf, self.centres[0:k])
                weights = np.min(distances, axis=1)
                prob_dist = weights / np.sum(weights)

                # select the next centroid from the data points such that the probability of choosing a point as centroid is
                # directly proportional to its distance from the nearest, previously chosen centroid

                i = choice(range(len(self.data_buf)), 1, p=prob_dist)
                self.centres[k, :] = self.data_buf[i]

    def iter(self):
        pass

    def converge(self):
        """
        loop over iter function until convergence
        :return:
        """

        iter_count = 0
        diff = None
        count = None
        cohesion = None
        time_diff = datetime.now()

        # continue to iterate as long as the number of iterations is below the maximum, and the change in centres is
        # greater than the tolerance
        while iter_count < self.max_iter and ((diff is None) or (diff > self.tol).any()):

            old_centroids = self.centres.copy()
            self.iter()

            count = np.zeros(shape=(len(self.centres), 1))
            cohesion = np.zeros(shape=(len(self.centres), 1))
            cluster_distances = self.get_distances()
            # get index of closest cluster for each data point
            k_min_indices = np.argmin(cluster_distances, axis=1)
            for i in range(len(self.data_buf)):
                count[k_min_indices[i]] += 1
                cohesion[k_min_indices[i]] += np.sum((self.data_buf[i] - self.centres[k_min_indices[i]]) ** 2)

            diff = paired_distances(old_centroids, self.centres, metric=self.distance_type)
            iter_count += 1

        time_diff -= datetime.now()

        # for each data point in data_buf, update the count and cohesion of the cluster it was added to
        self.count += count
        self.cohesion += cohesion

        for k in range(len(self.centres)):
            self.separation[k] = np.sum(self.count * np.sum((self.centres[k] - self.centres) ** 2)) / \
                                 np.sum(self.count)

        self.clustering_results.update(iter_count, time_diff)


class OriginalOnlineKMeans(OnlineK):

    def __init__(self, clustering_results, features, num_clusters, max_iter=10000, tol=0.0001, learning_rate=0.05,
                 distance_type='euclidean', init_type='random'):
        super().__init__(clustering_results, features, num_clusters, max_iter, tol, learning_rate, distance_type,
                         init_type)

    def iter(self):
        """
        perform a single iteration of online k-means convergence
        :return:
        """

        # calculate N * K sum of euclidean distance over all attributes from this point to all cluster centres
        cluster_distances = self.get_distances()

        # get index of closest cluster for each data point
        k_min_indices = np.argmin(cluster_distances, axis=1)

        for i in range(len(self.data_buf)):
            # update centre of cluster closest to this data point
            self.centres[k_min_indices[i]] += self.learning_rate * \
                                                (self.data_buf[i] - self.centres[k_min_indices[i]])


class HarmonicOnlineKMeans(OnlineK):

    def __init__(self, clustering_results, features, num_clusters, max_iter=10000, tol=0.0001, learning_rate=0.05,
                 distance_type='euclidean', init_type='random'):
        super().__init__(clustering_results, features, num_clusters, max_iter, tol, learning_rate, distance_type,
                         init_type)

    def iter(self):
        """
        perform a single iteration of online k-means convergence
        :return:
        """

        # calculate N * K sum of euclidean distance over all attributes from this point to all cluster centres

        for i in range(len(self.data_buf)):
            for k in range(self.num_clusters):
                cluster_distances = self.get_distances()
                # update this cluster centroid based on this data point
                self.centres[k] += self.learning_rate * (cluster_distances[i, k] ** (-4)) \
                                     * (np.sum([cluster_distances[i, kk] ** (-2)
                                                for kk in range(self.num_clusters)]) ** (-2)) \
                                     * (self.data_buf[i] - self.centres[k])


class InverseOnlineKMeans(OnlineK):

    def __init__(self, clustering_results, features, num_clusters, max_iter=10000, tol=0.0001, learning_rate=0.05,
                 distance_type='euclidean', init_type='random'):
        super().__init__(clustering_results, features, num_clusters, max_iter, tol, learning_rate, distance_type,
                         init_type)

    def iter(self):
        """
        perform a single iteration of online k-means convergence
        :return:
        """

        # calculate N * K sum of euclidean distance over all attributes from this point to all cluster centres
        cluster_distances = self.get_distances()

        # get indices of closest cluster
        k_min_indices = np.argmin(cluster_distances, axis=1)

        for i in range(len(self.data_buf)):

            n = 1
            self.centres[k_min_indices[i]] += self.learning_rate \
                                                * (((n + 1) * (cluster_distances[i, k_min_indices[i]] ** (n - 1)))
                                                   + (n * (cluster_distances[i, k_min_indices[i]] ** (n - 2))
                                                      * np.sum(
                                [cluster_distances[i, kk] for kk in range(self.num_clusters)
                                 if kk != k_min_indices[i]]))) \
                                                * (self.data_buf[i] - self.centres[k_min_indices[i]])

            for k in [kk for kk in range(self.num_clusters) if kk != k_min_indices[i]]:
                # TODO is it necessary to recalculate?
                cluster_distances = self.get_distances()
                self.centres[k] += self.learning_rate * ((cluster_distances[i, k_min_indices[i]] ** 2)
                                                           / cluster_distances[i, k]) \
                                     * (self.data_buf[i] - self.centres[k])


class OnlineFuzzyCMeans(OnlineK):

    def __init__(self, clustering_results, features, num_clusters, max_iter=10000, tol=0.0001, learning_rate=0.05,
                 distance_type='euclidean', init_type='random'):
        super().__init__(clustering_results, features, num_clusters, max_iter, tol, learning_rate, distance_type,
                         init_type)

        #  m > 1, fuzzyness parameter. the closer to m is to 1, the closer to hard kmeans,
        #  the greater m, the fuzzier (converge to the global cluster).
        self.m = m

    def iter(self):
        """
        perform a single iteration of online k-means convergence
        :return:
        """

        # calculate N * K sum of euclidean distance over all attributes from this point to all cluster centres

        # calculate N * K sum of euclidean distance over all attributes from this point to all cluster centres
        cluster_distances = self.get_distances()

        membership_matrix = np.sum([(cluster_distances / cluster_distances[:, kk]) ** (2 / (self.m - 1))
                                    for kk in range(self.num_clusters)]) ** (-1)

        for k in range(self.num_clusters):
            self.centres[k] = np.sum((membership_matrix[:, k] ** self.m) * self.data_buf) / \
                                np.sum(membership_matrix[:, k] ** self.m)


class OnlineKMedoids(OnlineK):

    def __init__(self, clustering_results, features, num_clusters, max_iter=10000, tol=0.0001, learning_rate=0.05,
                 distance_type='euclidean', init_type='random'):
        super().__init__(clustering_results, features, num_clusters, max_iter, tol, learning_rate, distance_type,
                         init_type)

    def iter(self):
        """
        perform a single iteration of online k-medioid convergence
        :return:
        """

        # calculate N * K sum of euclidean distance over all attributes from this point to all cluster centres
        cluster_distances = self.get_distances()

        # get index of closest cluster for each data point
        k_min_indices = np.argmin(cluster_distances, axis=1)

        """
        Calculate the dissimilarity matrix if it was not provided;
        3. Assign every object to its closest medioid;
        Swap phase:
        4. For each cluster check if any of the object of the cluster decreases the average dissimilarity coefficient; 
        if it does, select the entity that decreases this coefficient the most as the medoid for this cluster; 
        5. If at least one medoid has changed go to (3), else end the algorithm.
        """

        # for each new data point, check if a swap with the closest medioid would improve the that clusters stored metrics:
        # ie produce a more uniform distribution

        for i in range(len(self.data_buf)):
            # update centre of cluster closest to this data point
            self.centres[k_min_indices[i]] += self.learning_rate * \
                                                (self.data_buf[i] - self.centres[k_min_indices[i]])


class OnlineQualityThreshold:
    def __init__(self):
        pass


class ExpectationMaximization:
    def __init__(self):
        pass


class MeanShift:
    def __init__(self):
        pass


class LSHBC:
    """
    Locality Sensitive Hashing Based Clustering
    """

    def __init__(self):
        pass


class KWaySpectral:
    def __init__(self):
        pass

# class ELM(ClusteringAlgorithm):
#
#     def __init__(self, data_points):
#         super().__init__(data_points)
#
#
# class CODAS(ClusteringAlgorithm):
#
#     def __init__(self, data_points):
#         super().__init__(data_points)
#
#
# class DEC(ClusteringAlgorithm):
#
#     def __init__(self, data_points):
#         super().__init__(data_points)
#
#
# class CEDAS(ClusteringAlgorithm):
#
#     def __init__(self, data_points, micro_cluster_radius, clusters, fade_rate, min_cluster_threshold, outlier_samples,
#                  graph_struct):
#         super().__init__(data_points)
#         self.micro_cluster_radius = micro_cluster_radius
#         self.clusters = clusters
#         self.fade_rate = fade_rate
#         self.min_cluster_threshold = min_cluster_threshold
#         self.outlier_samples = outlier_samples
#         self.graph_struct = graph_struct
#
#     def start_cluster(self):
#         """
#
#         :return:
#         """
#         """
#         [d]=pdist2(O,O); % distance between all outliers
#         [Nin,NC]=max(sum(d<R)); % sum of number of samples within R for all data
#         if sum(d(:,NC)<R)>M
#         N=1; % new microC number
#         [In,~]=find(d(:,NC)<R); % find data within R
#         C=struct('C',mean(O(In,:),1)); % make centre mean of clustered data
#         C.L(N,:)=1; % Set life
#         C.T(N,:)=Nin; % Count number assigned
#         C.K(N,:)=sum(d(:,NC)<0.5*R); % Count number in kernel
#         G=addnode(G,1); % Add graph structure node
#         O(In,:)=[]; % Remove outliers that have been assigned to new microC
#         end
#         """
#         # get euclidean distance between all outlier_samples, where distance is less than micro cluster radius
#         dist = euclidean_distances(self.outlier_samples, self.outlier_samples)
#
#         # boolean matrix indicating if each distance is less than the micro cluster radius
#         is_shell_dist = dist < self.micro_cluster_radius
#
#         # number of sub micro-cluster radius distances for each data sample
#         num_samples = np.sum(is_shell_dist)
#
#         # number and index of data samples in new cluster corresponds to highest number of sub micro-cluster radius
#         # distances,
#         # if it passes min threshold
#         new_shell_count = np.max(num_samples)
#         new_micro_cluster_index = np.argmax(num_samples)
#
#         # if the number of data samples in this cluster is greater than the threshold, create a new cluster
#         if new_shell_count > self.min_cluster_threshold:
#
#             # new cluster mean
#             new_mean = np.mean(self.outlier_samples[is_shell_dist])[new_micro_cluster_index]
#
#             # boolean matrix indicating if each distance is less than the kernal cluster radius
#             is_kernal_dist = dist < (0.5 * self.micro_cluster_radius)
#             new_kernel_count = np.max(np.sum(is_kernal_dist))
#
#             # initialise new cluster with mean, life, total sample count and kernel sample count
#             self.clusters.append = Cluster(mean=new_mean, total_count=new_shell_count, life=1,
#                                            kernel_count=new_kernel_count)
#
#             for col in range(len(is_shell_dist[new_micro_cluster_index])):
#                 if is_shell_dist[new_micro_cluster_index][col]:
#                     self.outlier_samples = np.delete(self.outlier_samples,
#                                                      new_micro_cluster_index * self.outlier_samples.shape[1] + col)
#
#     def assign(self):
#         pass
#
#     def kill(self):
#         pass
#
#     def graph(self):
#         pass
#
#     def iter(self):
#         # if not micro-clusters exist, create the first
#         if len(self.clusters) == 0:
#             self.outlier_samples.append(self.data_pointss)
#             self.start_cluster()
#         # else assign and decay
#         else:
#             self.assign()
#             self.kill()
