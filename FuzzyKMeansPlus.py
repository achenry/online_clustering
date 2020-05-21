import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from random import randint, random


class FuzzyKMeansPlus:
    """
    Class defining KMeans++ Clustering Fit algorithm with variable fuzziness.
    """
    def __init__(self, n_clusters=8, init='kmeans++', fuzziness=2, max_iter=300, tol=1e-4, random_state=0):
        """
        initialise FuzzyKMeansPlus clustering fit object
        :param n_clusters: number of clusters to fit for
        :param init: either 'kmeans++' for re-initialisation of centroids, or a n_clusters * n_features ndarray of
                     centroids to use as initialisation
        :param fuzziness: fuzziness factor m, = 1 for hard clustering, approx 2 for softer clustering
        :param max_iter: maximum number of iterations to allow for covergence
        :param tol: tolearance of centroid change to allow for reaching convergence
        :param random_state:
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.fuzziness = fuzziness
        self.random_state = random_state

    def _e_step(self, X, centers, sample_weights):
        """
        Calculate fuzzy labels and labels for each given data point in X relative to centroids in centers,
        weighted by sample_weights
        :param X: N * n_features ndarray of data points to cluster
        :param centers: n_clusters * n_features ndarray of existing centroids
        :param sample_weights: N * 1 array of weights corresponding to each data point
        :return:
        """

        # for any data point which is the cluster centroid, this fuzzy_label should be one
        # as this data point certainly belongs to this cluster,
        # and all other fuzzy_labels in that data point row should be zero
        # for hard clustering,

        # get the distance between each data point - centroid pair
        euc_dist = euclidean_distances(X, centers, squared=True)
        # x = np.sum(euc_dist ** 2, axis=0)[:, np.newaxis]
        # x = np.argwhere(euclidean_distances(x, x) == 0)
        #
        # if np.any(np.logical_and(x[:, 0] == 27, x[:, 1] == 4)):
        #     print("here")

        argmin_euc_dist = np.argmin(euc_dist, axis=1)
        n_samples = X.shape[0]

        # for hard clustering, where each row of fuzzy_labels (for each data point) will have a single nonzero value
        # of 1 corresponding to the cluster to which that data point certainly belongs
        if self.fuzziness == 1:
            D = np.zeros((n_samples, self.n_clusters))
            for data_point_idx, closest_cluster_idx in enumerate(argmin_euc_dist):
                D[data_point_idx, closest_cluster_idx] = 1
        # for fuzzy clustering, the values over each row of fuzzy_labels will sum to 1
        else:
            with np.errstate(divide='ignore'):
                D = 1.0 / euc_dist

            x_eq_centroid_indices = np.argwhere(D == np.inf)
            D[x_eq_centroid_indices[:, 0], :] = 0
            D[x_eq_centroid_indices[:, 0], x_eq_centroid_indices[:, 1]] = 1

            D **= np.divide(1.0, (self.fuzziness - 1))
            D /= np.sum(D, axis=1)[:, np.newaxis]

        # n_samples * n_clusters ndarray of probability of membership of each sample to each cluster
        fuzzy_labels = D
        fuzzy_weights = sample_weights[:, np.newaxis] * (fuzzy_labels ** self.fuzziness)

        # x = np.sum(fuzzy_weights ** 2, axis=0)[:, np.newaxis]
        # for k in range(fuzzy_weights.shape[1] - 1):
        #     for kk in range(k + 1, fuzzy_weights.shape[1]):
        #         if np.all(np.abs(fuzzy_weights[:, k] - fuzzy_weights[:, kk]) < 10**(-20)):
        #             print(k, kk)
        #             break

        # n_samples * 1 array of index of cluster to which each sample most likely belongs
        labels = np.argmax(fuzzy_labels, axis=1)

        distances = np.sum((X - centers[labels]) ** 2, axis=1) ** 0.5

        inertia = np.sum(np.sum(fuzzy_weights * euc_dist, axis=1), axis=0)

        # inertia2 = 0
        # for k in range(self.n_clusters):
        #     for d in range(n_samples):
        #         inertia2 += fuzzy_weights[d, k] * np.sum((X[d] - centers[k]) ** 2)

        #inertia = np.sum(np.sum((X - centers[labels]) ** 2, axis=1), axis=0)
        # if np.abs(inertia1 - inertia) > 1e-4:
        #     print("here")

        return fuzzy_labels, fuzzy_weights, labels, distances, inertia

    def _m_step(self, X, fuzzy_weights, distances, sample_weights):
        """
        Calculate centroids based on given data points, the probabilities of each belonging to each cluster and
        sample weights
        :param X: N * n_features ndarray of data points
        :param fuzzy_weights: N * n_clusters weights of data samples as product of fuzziness and sample_weights
        :param sample_weights: N * 1 array of weights for each data point
        :return centers: n_clusters * n_features ndarray of centroids
        """

        # n_clusters x n_features ndarray of centroids are mean of weighted data points
        # n_samples, n_features = X.shape
        # centers = np.zeros((self.n_clusters, n_features))
        weight_in_cluster = np.sum(fuzzy_weights, axis=0)
        empty_clusters = np.where(weight_in_cluster == 0)[0]

        if len(empty_clusters):
            # find points to reassign empty clusters to

            # indices of data points with longest to shortest distances to closest centroid
            far_from_centers = distances.argsort()[::-1]

            # loop through clusters which are too far from all data points
            for d, cluster_id in enumerate(empty_clusters):
                # two relocated clusters could be close to each other

                # set the centroid of this empty cluster to the datapoint furthest from its closest centroid
                # corresponding fuzzy_label sum is 0, so revert to sample_weights only
                far_index = far_from_centers[d]
                # if not np.any(np.all([X[far_index] == center for center in ]))
                fuzzy_weights[far_index, cluster_id] = sample_weights[far_index]
                weight_in_cluster[cluster_id] = np.sum(fuzzy_weights[:, cluster_id], axis=0)
                # new_center = X[far_index] * sample_weights[far_index]
                # centers[cluster_id] = new_center
                # weight_in_cluster[cluster_id] = sample_weights[far_index]

        # for d in range(n_samples):
        #     # center is weighted mean of all data points and the weight of those data points to that cluster
        #     centers[labels[d]] += X[d] * weights[d, labels[d]]

        # center is weighted mean of all data points and the weight of those data points to that cluster
        centers = np.dot(X.T, fuzzy_weights).T
        centers /= weight_in_cluster[:, np.newaxis]

        if np.any(np.isnan(centers)) or np.any(np.isinf(centers)):
            # if a centroid has no data points which are closest to it because n_samples < n_clusters
            print('nan centers')

        # for k in range(len(centers) - 1):
        #     for kk in range(k + 1, len(centers)):
        #         if np.all(centers[k] == centers[kk]):
        #             print("here")
        #             break

        return centers

    def fit(self, X, y=None, sample_weights=None):
        """
        Fit the given data points to clusters based on self.init ndarray or by kmeans++ initialisation
        :param X: N * n_features ndarray of data points
        :param y: none, left by convention
        :param sample_weights: N * 1 array of weights for each data point
        :return: self
        """

        # declare fuzzy_labels, labels arrays
        # fuzzy_labels, labels = None, None
        best_labels, best_fuzzy_labels, best_inertia, best_centers = None, None, None, None

        # declare data dimensions
        n_samples, n_features = X.shape

        # if no sample weight is given, assign a weight of 1 to all data points
        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        # if centroids are to be re-initialised, do so by kmeans++
        if type(self.init) is str and self.init == "kmeans++":
            # https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
            # https://github.com/pavankalyan1997/Machine-learning-without-any-libraries/blob/master/2.Clustering/1.K_Means_Clustering/Kmeans.py

            # Randomly select the first cluster center from the data points and append it to the centroid matrix.
            i = randint(0, n_samples - 1)
            centers = np.array([X[i]])

            # for the remaining cluster centroids
            for k in range(1, self.n_clusters):

                # calculate the minimum euclidean distance square from the already chosen centroids to each data point
                minimum_distances = np.min(euclidean_distances(X, centers, squared=True), axis=1)

                # calculate the probabilities of choosing each data point as the next centroid
                prob = (minimum_distances / np.sum(minimum_distances)) * sample_weights

                # calculate the cumulative probability
                cumulative_prob = np.cumsum(prob)

                # select a random number between 0 to 1, get the index (i) of the cumulative probability distribution
                # which is just greater than the chosen random number and assign the data point corresponding to the
                # selected index (i) as the new centroid.
                r = random()
                i = 0
                for j, cp in enumerate(cumulative_prob):
                    if r < cp:
                        i = j
                        break
                centers = np.append(centers, [X[i]], axis=0)
        # otherwise existing centroids are passed as initialisation
        else:
            centers = self.init

        if np.any(np.isnan(centers)) or np.any(np.isinf(centers)):
            print("here")

        # apply kmeans fit with sample-weights for existing centroids and new data points
        # https://github.com/leapingllamas/medium_posts/blob/master/observation_weighted_kmeans/data_weighted_kmeans.py

        # for hard clustering:
        if self.fuzziness == 1:
            fuzzy_labels, fuzzy_weights, labels, distances, inertia = self._e_step(X, centers, sample_weights)
        # for fuzzy clustering:
        else:
            random_state = check_random_state(self.random_state)
            fuzzy_labels = random_state.rand(n_samples, self.n_clusters)
            fuzzy_labels /= fuzzy_labels.sum(axis=1)[:, np.newaxis]
            fuzzy_weights = sample_weights[:, np.newaxis] * (fuzzy_labels ** self.fuzziness)
            labels = np.argmax(fuzzy_labels , axis=1)

            distances = np.sum((X - centers[labels]) ** 2, axis=1) ** 0.5

        centers = self._m_step(X, fuzzy_weights, distances, sample_weights)

        for i in range(self.max_iter):
            centers_old = centers.copy()

            # get new labels = membership of each data point to each cluster
            fuzzy_labels, fuzzy_weights, labels, distances, inertia = self._e_step(X, centers, sample_weights)

            # get new centers = weighted mean of each cluster of data points
            centers = self._m_step(X, fuzzy_weights, distances, sample_weights)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_fuzzy_labels = fuzzy_labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia

            # if magnitude change in centers is insignificant, break convergence loop
            center_shift = np.ravel(centers_old - centers, order='K')
            # The Euclidean norm when shift is a vector (single feature),
            # the Frobenius norm when shift is a matrix (> 1 feature)
            center_shift_total = np.dot(center_shift, center_shift)
            if center_shift_total <= self.tol:
                break

        if center_shift_total > 0:
            best_fuzzy_labels, _, best_labels, _, best_inertia = self._e_step(X, best_centers, sample_weights)

        # set the attributes of self to the final solution
        self.fuzzy_labels_ = best_fuzzy_labels
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.n_iter_ = i + 1
        self.inertia_ = best_inertia

        return self

