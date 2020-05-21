import warnings
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed, effective_n_jobs

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import row_norms
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
import sklearn.cluster._kmeans
from sklearn.cluster._kmeans import (_check_normalize_sample_weight, _init_centroids, _labels_inertia,
                                     check_random_state, _k_means, squared_norm, sp, check_array, _num_samples,
                                     _tolerance, _validate_center_shape, k_means_elkan)

def m_step(X, fuzzy_labels, m):

    weights = fuzzy_labels ** m

    # shape: n_clusters x n_features
    centers = np.dot(weights, X)
    centers /= weights.sum(axis=1)[:, np.newaxis]
    return centers

def e_step(X, centers, m):
    D = 1.0 / euclidean_distances(X, centers, squared=True)
    D **= 1.0 / (m - 1)
    D /= np.sum(D, axis=1)[:, np.newaxis]

    # shape: n_samples x k
    fuzzy_labels = D
    labels = fuzzy_labels.argmax(axis=1)

    return fuzzy_labels, labels

def _fuzzykmeans_single_elkan(X, m, sample_weight, n_clusters, max_iter=300,
                         init='k-means++', verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4,
                         precompute_distances=True):
    if sp.issparse(X):
        raise TypeError("algorithm='elkan' not supported for sparse input X")

    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)

    fuzzy_labels = random_state.rand(n_samples, n_clusters)
    fuzzy_labels /= fuzzy_labels.sum(axis=1)[:, np.newaxis]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)
    centers = m_step(centers, fuzzy_labels, m)
    centers = np.ascontiguousarray(centers)
    if verbose:
        print('Initialization complete')

    checked_sample_weight = _check_normalize_sample_weight(sample_weight, X)
    centers, labels, n_iter = k_means_elkan(X, checked_sample_weight,
                                            n_clusters, centers, tol=tol,
                                            max_iter=max_iter, verbose=verbose)
    fuzzy_labels, labels = e_step(X, centers, m)
    centers = m_step(X, fuzzy_labels, m)

    if sample_weight is None:
        inertia = np.sum((X - centers[labels]) ** 2, dtype=np.float64)
    else:
        sq_distances = np.sum((X - centers[labels]) ** 2, axis=1,
                              dtype=np.float64) * checked_sample_weight
        inertia = np.sum(sq_distances, dtype=np.float64)
    return fuzzy_labels, labels, inertia, centers, n_iter



def _fuzzykmeans_single_lloyd(X, m, sample_weight, n_clusters, max_iter=300,
                         init='k-means++', verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4,
                         precompute_distances=True):
    """A single run of k-means, assumes preparation completed prior.

    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    tol : float, optional
        The relative increment in the results before declaring convergence.

    verbose : boolean, optional
        Verbosity mode

    x_squared_norms : array
        Precomputed x_squared_norms.

    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)

    fuzzy_labels = random_state.rand(n_samples, n_clusters)
    fuzzy_labels /= fuzzy_labels.sum(axis=1)[:, np.newaxis]

    sample_weight = _check_normalize_sample_weight(sample_weight, X)

    best_fuzzy_labels, best_labels, best_inertia, best_centers = None, None, None, None
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)

    centers = m_step(centers, fuzzy_labels, m)

    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, sample_weight, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
        fuzzy_labels, labels = e_step(X, centers, m)

        # computation of the means is also called the M-step of EM
        # if sp.issparse(X):
        #     centers = _k_means._centers_sparse(X, sample_weight, labels,
        #                                        n_clusters, distances)
        # else:
        #     centers = _k_means._centers_dense(X, sample_weight, labels,
        #                                       n_clusters, distances)
        centers = m_step(X, fuzzy_labels, m)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_fuzzy_labels = fuzzy_labels.copy()
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, sample_weight, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
        best_fuzzy_labels, best_labels = e_step(X, centers, m)

    return best_fuzzy_labels, best_labels, best_inertia, best_centers, i + 1


# sklearn.cluster._kmeans._kmeans_single_lloyd = _fuzzykmeans_single_lloyd


class FuzzyKMeansPlus(FuzzyKMeans, KMeans):
    def __init__(self, n_clusters=8, m=1, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):

        FuzzyKMeans.__init__(self, k=n_clusters, m=m, max_iter=max_iter, random_state=random_state, tol=tol)
        KMeans.__init__(self, n_clusters=n_clusters, init=init, n_init=n_init,
                        max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,
                        verbose=verbose, random_state=random_state, copy_x=copy_x,
                        n_jobs=n_jobs, algorithm=algorithm)

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        self
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        n_init = self.n_init
        if n_init <= 0:
            raise ValueError("Invalid number of initializations."
                             " n_init=%d must be bigger than zero." % n_init)

        if self.max_iter <= 0:
            raise ValueError(
                'Number of iterations should be a positive number,'
                ' got %d instead' % self.max_iter
            )

        # avoid forcing order when copy_x=False
        order = "C" if self.copy_x else None
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32],
                        order=order, copy=self.copy_x)
        # verify that the number of samples given is larger than k
        if _num_samples(X) < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                _num_samples(X), self.n_clusters))

        tol = _tolerance(X, self.tol)

        # If the distances are precomputed every job will create a matrix of
        # shape (n_clusters, n_samples). To stop KMeans from eating up memory
        # we only activate this if the created matrix is guaranteed to be
        # under 100MB. 12 million entries consume a little under 100MB if they
        # are of type double.
        precompute_distances = self.precompute_distances
        if precompute_distances == 'auto':
            n_samples = X.shape[0]
            precompute_distances = (self.n_clusters * n_samples) < 12e6
        elif isinstance(precompute_distances, bool):
            pass
        else:
            raise ValueError(
                "precompute_distances should be 'auto' or True/False"
                ", but a value of %r was passed" %
                precompute_distances
            )

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype.type, copy=True)
            _validate_center_shape(X, self.n_clusters, init)

            if n_init != 1:
                warnings.warn(
                    'Explicit initial center position passed: '
                    'performing only one init in k-means instead of n_init=%d'
                    % n_init, RuntimeWarning, stacklevel=2)
                n_init = 1

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        best_labels, best_inertia, best_centers = None, None, None
        algorithm = self.algorithm
        if self.n_clusters == 1:
            # elkan doesn't make sense for a single cluster, full will produce
            # the right result.
            algorithm = "full"
        if algorithm == "auto":
            algorithm = "full" if sp.issparse(X) else 'elkan'
        if algorithm == "full":
            kmeans_single = _fuzzykmeans_single_lloyd
        elif algorithm == "elkan":
            kmeans_single = _fuzzykmeans_single_elkan
        else:
            raise ValueError("Algorithm must be 'auto', 'full' or 'elkan', got"
                             " %s" % str(algorithm))

        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        if effective_n_jobs(self.n_jobs) == 1:
            # For a single thread, less memory is needed if we just store one
            # set of the best results (as opposed to one set per run per
            # thread).
            for seed in seeds:
                # run a k-means once
                fuzzy_labels, labels, inertia, centers, n_iter_ = kmeans_single(
                    X, self.m, sample_weight, self.n_clusters,
                    max_iter=self.max_iter, init=init, verbose=self.verbose,
                    precompute_distances=precompute_distances, tol=tol,
                    x_squared_norms=x_squared_norms, random_state=seed)
                # determine if these results are the best so far
                if best_inertia is None or inertia < best_inertia:
                    best_fuzzy_labels = fuzzy_labels.copy()
                    best_labels = labels.copy()
                    best_centers = centers.copy()
                    best_inertia = inertia
                    best_n_iter = n_iter_
        else:
            # parallelisation of k-means runs
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(kmeans_single)(
                    X, self.m, sample_weight, self.n_clusters,
                    max_iter=self.max_iter, init=init,
                    verbose=self.verbose, tol=tol,
                    precompute_distances=precompute_distances,
                    x_squared_norms=x_squared_norms,
                    # Change seed to ensure variety
                    random_state=seed
                )
                for seed in seeds)
            # Get results with the lowest inertia
            fuzzy_labels, labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            best_fuzzy_labels = fuzzy_labels[best]
            best_labels = labels[best]
            best_inertia = inertia[best]
            best_centers = centers[best]
            best_n_iter = n_iters[best]

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2
            )

        self.cluster_centers_ = best_centers
        self.fuzzy_labels_ = best_fuzzy_labels
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self
