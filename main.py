from input_data_reader import InputDataReader
from input_parameter_reader import InputParameterReader
from clustering_algorithms2 import Feature, OnlineKMeansPlus, OnlineOptimalKMeansPlus, OfflineOptimalKMeansPlus
from clustering_results2 import ClusteringResults
import numpy as np
import os
from datetime import datetime
import multiprocessing as mp
import pandas as pd


def main():
    # open shelve
    # shelf = shelve.open('db')

    # make required results directory
    if not os.path.exists("./results"):
        os.mkdir("./results")

    scenarios = {'Offline Optimal-K Crisp KMeans++': {'algorithm': 2, 'alpha': 0.05, 'fuzziness': 1},  # 0
                 'Offline Optimal-K Fuzzy KMeans++': {'algorithm': 2, 'alpha': 0.05, 'fuzziness': 2},  # 1

                 'Online Periodic-K Crisp KMeans++': {'algorithm': 0, 'fuzziness': 1, 'init_num_clusters': 6},  # 2
                 'Online Periodic-K Fuzzy KMeans++': {'algorithm': 0, 'fuzziness': 2, 'init_num_clusters': 6},  # 3

                 'Online Optimal-K Crisp KMeans++ Window-Size 2': {'algorithm': 1, 'window_size': 2, 'alpha': 0.05,  # 4
                                                                   'fuzziness': 1,
                                                                   'init_num_clusters': 4},
                 'Online Optimal-K Crisp KMeans++ Window-Size 4': {'algorithm': 1, 'window_size': 4, 'alpha': 0.05,  # 5
                                                                   'fuzziness': 1,
                                                                   'init_num_clusters': 4},
                 'Online Optimal-K Crisp KMeans++ Window-Size 8': {'algorithm': 1, 'window_size': 8, 'alpha': 0.05,  # 6
                                                                   'fuzziness': 1,
                                                                   'init_num_clusters': 4},

                 'Online Optimal-K Fuzzy KMeans++ Window-Size 2': {'algorithm': 1, 'window_size': 2, 'alpha': 0.05,  # 7
                                                                    'fuzziness': 2,
                                                                   'init_num_clusters': 4},
                 'Online Optimal-K Fuzzy KMeans++ Window-Size 4': {'algorithm': 1, 'window_size': 4, 'alpha': 0.05,  # 8
                                                                    'fuzziness': 2,
                                                                   'init_num_clusters': 4},
                 'Online Optimal-K Fuzzy KMeans++ Window-Size 8': {'algorithm': 1, 'window_size': 8, 'alpha': 0.05,  # 9
                                                                    'fuzziness': 2,
                                                                   'init_num_clusters': 4},
                 }

    scenario_indices = [0, 1, 2, 3, 5, 8]
    new_scenarios = {}
    for scenario_index in scenario_indices:
        key = list(scenarios)[scenario_index]
        new_scenarios[key] = scenarios[key]
    scenarios = new_scenarios

    # multiprocessing pool
    pool = mp.Pool(mp.cpu_count())

    for scenario_name, scenario_params in scenarios.items():

        # shelf[scenario_name] = {}

        print(f"\nStarting scenario {scenario_name}.")

        print(f"\nReading input parameters.")

        # initialise Input Parameter Reader object
        input_parameter_reader = InputParameterReader()

        # read input parameters from input_params.ini
        input_parameter_reader.read_input_parameters()

        input_params = input_parameter_reader.input_params

        # update parameters depending on scenario
        input_params.update([('test_name', scenario_name)])
        input_params.update([(k, v) for k, v in scenario_params.items()])

        data_type = input_params['data_type']
        batch_size = input_params['batch_size']
        init_batch_size = input_params['init_batch_size']
        init_num_clusters = input_params['init_num_clusters']
        tol = input_params['tol']
        max_iter = input_params['max_iter']
        time_decay_const = input_params['time_decay_const']
        fuzziness = input_params['fuzziness']
        alpha = input_params['alpha']
        algorithm = input_params['algorithm']
        num_samples_to_run = input_params['num_samples_to_run']
        window_size = input_params['window_size']
        test_name = input_params['test_name']
        plotting_data_step = input_params['plotting_data_step']
        csv_path = input_params['csv_path']
        customer_ids = input_params['customer_ids']
        feature_names = input_params['feature_names']

        print("\nInput parameters read.")

        print(f"\nCreating results directories.")

        # if this test run results folder has not yet been created, create it
        if not os.path.exists(f"./results/{test_name}"):
            os.mkdir(f"./results/{test_name}")

        print(f"\nResults directories created.")

        print(f"\nReading data stream.")

        # initialise Data Stream Reader
        data_stream_reader = InputDataReader()

        # read input data stream from csv file
        # if reading load_pattern data, where features correspond to half-hour time intervals
        if data_type == 0:
            data_stream_reader.read_load_pattern_data(csv_path)
            # features = [Feature('Energy_' + str(hour)) for hour in np.arange(0, 24, 0.5)]
        # elif reading load time-series data, where features correspond to energy and timestamp components
        elif data_type == 1:
            data_stream_reader.read_load_time_series_data(csv_path, customer_ids)

        print(f"\nData stream read.")

        # initialise Feature objects
        features = [Feature(fn) for fn in feature_names]

        # initialise the Clustering Results object
        clustering_results = ClusteringResults(test_name)

        # if performing an Online Periodic K-Means++ clustering
        if algorithm == 0:
            clustering_algorithm_class = OnlineKMeansPlus
            clustering_algorithm_args = {'features': features, 'init_num_clusters': init_num_clusters,
                                         'batch_size': batch_size, 'init_batch_size': init_batch_size,
                                         'max_iter': max_iter, 'tol': tol, 'fuzziness': fuzziness}
        # else if performing an Online Optimal KMeans++ clustering
        elif algorithm == 1:
            clustering_algorithm_class = OnlineOptimalKMeansPlus
            clustering_algorithm_args = {'features': features, 'init_num_clusters': init_num_clusters,
                                         'batch_size': batch_size, 'init_batch_size': init_batch_size,
                                         'time_decay_const': time_decay_const, 'fuzziness': fuzziness,
                                         'alpha': alpha, 'max_iter': max_iter, 'tol': tol, 'window_size': window_size}

        # else if performing an Offline Optimal KMeans++ clustering
        elif algorithm == 2:
            clustering_algorithm_class = OfflineOptimalKMeansPlus
            clustering_algorithm_args = {'features': features, 'init_num_clusters': init_num_clusters,
                                         'batch_size': batch_size, 'init_batch_size': init_batch_size,
                                         'time_decay_const': time_decay_const,
                                         'fuzziness': fuzziness, 'alpha': alpha, 'max_iter': max_iter, 'tol': tol}

        for cid in customer_ids:

            print(f"\nStarting customer {cid}.")

            # get data from chosen customers and features
            data_stream = data_stream_reader.data.loc[data_stream_reader.data.CustomerID == cid, feature_names]

            # choose the last index of the data to read to. Results will be formulated based on the clustering solution
            # at this index
            if type(num_samples_to_run) is int:
                if num_samples_to_run < 0:
                    final_idx = len(data_stream.index) + num_samples_to_run
                elif num_samples_to_run > len(data_stream.index):
                    final_idx = len(data_stream.index)
                else:
                    final_idx = num_samples_to_run
            else:
                final_idx = len(data_stream.index)

            # if this clustering has already been run, just fetch the data and plot results at the given index
            # plot results and output csv results at this index
            calculate_results = False
            if not os.path.exists(f"./results/{test_name}/input_params.csv"):
                calculate_results = True
            else:
                existing_run_params = pd.read_csv(f"./results/{test_name}/input_params.csv")
                if cid in existing_run_params.customer_ids \
                        and existing_run_params.loc[existing_run_params.customer_ids == cid, 'num_samples_to_run'].iloc[
                    0] >= final_idx:

                    clustering_results.finalise_at_index(cid, data_stream.iloc[0:final_idx], features)
                    clustering_results.plot_cluster_evolution(1, cid, show=False, last_fig_only=True,
                                                              export_all=True if (final_idx / plotting_data_step) < 25 else False)
                    clustering_results.plot_cluster_metric_evolution(1, cid, show=False, last_fig_only=True,
                                                                             export_all=True if (final_idx / plotting_data_step) < 25 else False)
                else:
                    calculate_results = True

            if calculate_results:

                if not os.path.exists(f"./results/{test_name}/{cid}"):
                    os.mkdir(f"./results/{test_name}/{cid}")

                # reset evolutionary results
                clustering_results.reset()

                # re-initialise clustering algorithm object
                clustering_algorithm = clustering_algorithm_class(**clustering_algorithm_args)

                # if performing an online algorithm
                is_online_algorithm = True if algorithm in [0, 1] else False

                total_running_time = datetime.now()

                if is_online_algorithm:

                    cluster_set = None
                    for i in range(final_idx):
                        print(f"\nSample count {i}")

                        # feed incoming data to algorithm object one sample at a time
                        clustering_algorithm.feed_data(data_stream.iloc[i].values)

                        # fetch resulting cluster set object and convergence metrics
                        cluster_set, running_time, num_iterations, inertia, compactness, dbi \
                            = clustering_algorithm.update_clusters(i, pool)

                        # update results of this test
                        clustering_results.update(sample_count=i, num_clusters=cluster_set.num_clusters,
                                                  centroids=cluster_set.centroids, running_time=running_time,
                                                  num_iterations=num_iterations,
                                                  masses=cluster_set.masses,
                                                  inertia=inertia,
                                                  compactness=compactness,
                                                  dbi=dbi,
                                                  inertias=np.sum(cluster_set.errors_squared, axis=1),
                                                  st_centroids=cluster_set.st_centroids,
                                                  alpha=clustering_algorithm.alpha)

                    # perform final crisp clustering
                    # TODO this should only merge close clusters. Generate bootstrapped datasets around clusters
                    #  or just merge significantly close clusters...should already have happened!
                    # final_clustering_algorithm_class = OfflineOptimalKMeansPlus
                    # final_clustering_algorithm_args = {'features': features, 'init_num_clusters': init_num_clusters,
                    #                                    'batch_size': batch_size, 'init_batch_size': init_batch_size,
                    #                                    'gravitational_const': gravitational_const,
                    #                                    'time_decay_const': time_decay_const,
                    #                                    'fuzziness': 1, 'alpha': alpha, 'max_iter': max_iter, 'tol': tol,
                    #                                    'cluster_set': cluster_set}
                    # final_clustering_algorithm = final_clustering_algorithm_class(**final_clustering_algorithm_args)
                    # # n_samples_per_cluster = np.asarray(cluster_set.masses, dtype='int')
                    # # bootstrapped_data, _ = synthetic_dataset, _ = make_blobs(n_samples=n_samples_per_cluster,
                    # #                                                          n_features=len(features),
                    # #                                                          centers=cluster_set.centroids, shuffle=False)
                    # final_clustering_algorithm.feed_data(cluster_set.centroids, cluster_set.masses)
                    # cluster_set, running_time, num_iterations, sse, compactness, dbi \
                    #     = final_clustering_algorithm.update_clusters(pool)
                    # clustering_results.update(sample_count=i, num_clusters=cluster_set.num_clusters,
                    #                           centroids=cluster_set.centroids, running_time=running_time,
                    #                           num_iterations=num_iterations, online_sse=sse,
                    #                           online_compactness=compactness, online_dbi=dbi,
                    #                           masses=cluster_set.masses,
                    #                           sses=cluster_set.sses,
                    #                           st_centroids=cluster_set.st_centroids,
                    #                           alpha=clustering_algorithm.alpha,
                    #                           gravitational_const=clustering_algorithm.gravitational_const)

                    # clustering_algorithm.close_pool()

                # else if performing an offline algorithm
                else:

                    input_params['batch_size'] = final_idx

                    # feed data to algorithm object
                    # data = data_stream.loc[:final_idx, [feat.name for feat in features]].dropna().values
                    print(f"\nFeeding data stream.")
                    clustering_algorithm.feed_data(data_stream.iloc[0:final_idx].values, np.ones(final_idx))

                    # fetch resulting cluster set object and convergence metrics
                    print(f"\nFitting clusters.")
                    cluster_set, running_time, num_iterations, inertia, compactness, dbi \
                        = clustering_algorithm.update_clusters(pool)

                    # update results of this test
                    print(f"\nUpdating results.")
                    clustering_results.update(sample_count=0, num_clusters=cluster_set.num_clusters,
                                              centroids=cluster_set.centroids, running_time=running_time,
                                              num_iterations=num_iterations,
                                              masses=cluster_set.masses,
                                              inertia=inertia,
                                              compactness=compactness,
                                              dbi=dbi,
                                              inertias=np.sum(cluster_set.errors_squared, axis=1),
                                              st_centroids=cluster_set.st_centroids,
                                              alpha=clustering_algorithm.alpha)

                total_running_time = (datetime.now() - total_running_time).total_seconds()

                # Calculate resulting metrics and Save to File with description of parameters
                print("\nClustering finished. Exporting .csv files.")
                clustering_results.finalise(cid, input_params, features, data_stream.iloc[0:final_idx], cluster_set,
                                            total_running_time)

                # Plot Results and Save to File with description of parameters
                print("\nResults exported. Exporting plots.")
                clustering_results.plot_cluster_evolution(plotting_data_step, cid, show=False,
                                                          last_fig_only=False if is_online_algorithm else True,
                                                          export_all=True if (final_idx / plotting_data_step) < 25 else False)
                clustering_results.plot_cluster_metric_evolution(plotting_data_step, cid, show=False,
                                                                 last_fig_only=False if is_online_algorithm else True,
                                                                 export_all=True if (final_idx / plotting_data_step) < 25 else False,
                                                                 actual_metrics=True)
                clustering_results.plot_parameter_evolution(plotting_data_step, cid, show=False)

                # shelf[scenario_name][cid] = clustering_results

                print("\nPlots exported. Clustering run finished.")

    pool.close()

    # shelf.close()


if __name__ == '__main__':
    main()
