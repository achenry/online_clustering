from input_data_reader import InputDataReader
from input_parameter_reader import InputParameterReader
from clustering_algorithms2 import Feature, KMeansPlus, GravitationalKMeansPlus
from clustering_results2 import ClusteringResults
import numpy as np


def main():

    input_parameter_reader = InputParameterReader()
    input_parameter_reader.read_input_parameters()
    data_type = input_parameter_reader.data_type
    batch_size = input_parameter_reader.batch_size
    init_batch_size = input_parameter_reader.init_batch_size
    init_num_clusters = input_parameter_reader.init_num_clusters
    tol = input_parameter_reader.tol
    max_iter = input_parameter_reader.max_iter
    gravitational_const = input_parameter_reader.gravitational_const  # * 10**(-11)
    k_change_threshold = input_parameter_reader.k_change_threshold
    algorithm = input_parameter_reader.algorithm
    quick_test = input_parameter_reader.quick_test
    test_name = input_parameter_reader.test_name
    test_params = input_parameter_reader.test_params

    data_stream_reader = InputDataReader()

    # if reading load_pattern data, where features correspond to half-hour time intervals
    if data_type == 0:
        data_stream_reader.read_load_pattern_data()
        data_stream = data_stream_reader.data
        features = [Feature('Energy_' + str(hour)) for hour in np.arange(0, 24, 0.5)]
    # elif reading load time-series data, where features correspond to power value and timestamp components
    elif data_type == 1:
        data_stream_reader.read_load_time_series_data()
        data_stream = data_stream_reader.data[['Month', 'DayOfWeek', 'Hour', 'Energy']]
        features = [Feature('Hour', lb=0, ub=24, step=0.5), Feature('Energy')]
        # features = [Feature('Energy')]


    clustering_results = ClusteringResults(data_stream, features)

    # Online Periodic K-Means++
    if algorithm == 0:
        clustering_algorithm = KMeansPlus(features, init_num_clusters, batch_size, init_batch_size, window_size=100)
    # Online gravitational optimal Kmeans++
    elif algorithm == 1:
        # Gravitational SSE K-Means++
        clustering_algorithm = GravitationalKMeansPlus(features,
                                                       init_num_clusters=init_num_clusters,
                                                       batch_size=batch_size,
                                                       init_batch_size=init_batch_size,
                                                       outlier_threshold=k_change_threshold,
                                                       gravitational_const=gravitational_const,
                                                       window_size=100)

    # Feed Data Stream to Algorithm
    cluster_set = np.array([])
    final_idx = int(0.05 * len(data_stream.index)) if quick_test else len(data_stream.index)
    for i in range(0, final_idx):

        # feed incoming data to algorithm
        clustering_algorithm.feed_data(data_stream.iloc[i])

        # fetch output cluster and processing metrics from algorithm
        cluster_set, running_time, num_iterations, sse = clustering_algorithm.update_clusters(i)

        # update results of this test
        clustering_results.update(convergence_count=i, num_clusters=cluster_set.num_clusters,
                                  centroids=cluster_set.centroids, running_time=running_time,
                                  num_iterations=num_iterations, online_sses=cluster_set.sses, masses=cluster_set.masses,
                                  counts=cluster_set.counts,
                                  centre_of_masses=cluster_set.centre_of_masses)

    # Finalise Results
    clustering_results.finalise(test_name, cluster_set)

    # Plot Results and Save to File with description of parameters
    clustering_results.plot_cluster_evolution()

    # Print Results and Save to File with description of parameters
    clustering_results.results.to_csv('KMP' + '_clustering_results')
    clustering_results.clusters.to_csv('KMP' + '_clusters')


if __name__ == '__main__':
    main()
