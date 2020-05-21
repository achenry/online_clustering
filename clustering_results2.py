import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
import numpy as np
from numpy.linalg import norm
import os
import re

class ClusteringResults:
    """
    Class defining resulting cluster attributes and clustering metrics from a full run of a Clustering Algorithm over
    all the given data
    """

    def __init__(self, test_name):
        """
        initialise the ClusteringResults object
        :param test_name: Name of test run and folder in which all csv files and plots will be stored in
                  ie in ./results/test_name
        """
        self.test_name = test_name

        if os.path.exists(f'./results/{self.test_name}/input_params.csv'):
            self.input_params = pd.read_csv(f'./results/{self.test_name}/input_params.csv')
        # else, initialise a new dataframe
        else:
            self.input_params = pd.DataFrame()

        # clusters.csv stores the precise centroid, standard deviation, count, sse, compactness and dbi values
        # of each cluster of each algorithm run,
        # once the algorithm has run through all given data points (up to final_idx in main.py)
        # SSE, Standard Deviation, Count, Compactness and DBI are calculated accurately afterwards based on the actual
        # dataset, rather then the approximated metrics calculated in the online mode
        # if an output clusters.csv file already exists, read it
        if os.path.exists(f'./results/{self.test_name}/clusters.csv'):
            self.clusters = pd.read_csv(f'./results/{self.test_name}/clusters.csv')
        # else, initialise a new dataframe
        else:
            self.clusters = pd.DataFrame(
                columns=['CustomerID', 'ClusterNumber', 'Centroid', 'Standard Deviation', 'Count',
                         'SSE', 'Compactness', 'DBI', 'NumberConvergences'])

        # final_results.csv stores the mean compactness of all clusters, mean dbi of all clusters, separation, total
        # sse of all clusters, average running time for each convergence, average number of iterations for each
        # convergence, number of convergences (approximately number of batched of data delivered to algorithm),
        # number of clusters once the algorithm has run through all given data points (up to final_idx in main.py)
        # if an output final_results.csv file already exists, read it
        if os.path.exists(f'./results/final_results.csv'):
            self.final_results = pd.read_csv(f'./results/final_results.csv')
        # else, initialise a new dataframe
        else:
            self.final_results = pd.DataFrame(
                columns=['CustomerID', 'Compactness', 'DBI', 'Separation', 'SSE', 'AverageRunningTime',
                         'AverageNumberIterations', 'NumberConvergences', 'NumberClusters', 'AbsoluteRunningTime',
                         'RelativeRunningTime', 'ScenarioName'])

        # evolutionary_results.csv stores the evolving number of clusters, number of iterations, running time for, 
        # sse sum /compactness mean/dbi mean since last initialise_clusters was last called for each convergence
        # if an output evolutionary_results.csv file already exists, read it
        if os.path.exists(f'./results/{self.test_name}/evolutionary_results.csv'):
            self.evolutionary_results = pd.read_csv(f'./results/{self.test_name}/evolutionary_results.csv')
        # else, initialise a new dataframe
        else:
            self.evolutionary_results = pd.DataFrame(columns=['CustomerID', 'ConvergenceCount', 'NumberClusters',
                                                              'NumberIterations', 'RunningTime', 'SSESinceInit',
                                                              'CompactnessSinceInit', 'DBISinceInit', 'Centroids'])

        self.sample_count = []
        self.num_clusters = []
        self.centroids = []
        self.running_time = []
        self.num_iterations = []
        self.online_sse = []
        self.online_compactness = []
        self.online_dbi = []
        self.masses = []
        self.counts = []
        self.probabilistic_counts = []
        self.centre_of_masses = []
        self.alpha = []
        self.gravitational_const = []

        self.data = None

    def reset(self):
        self.sample_count = []
        self.num_clusters = []
        self.centroids = []
        self.running_time = []
        self.num_iterations = []
        self.online_sse = []
        self.online_compactness = []
        self.online_dbi = []
        self.masses = []
        self.counts = []
        self.probabilistic_counts = []
        self.centre_of_masses = []
        self.alpha = []
        self.gravitational_const = []

    def update(self, sample_count, num_clusters, centroids, running_time, num_iterations, online_sse,
               online_compactness, online_dbi, masses,
               counts, probabilistic_counts, centre_of_masses, alpha, gravitational_const):
        """
        This method is called each time after data is fed to Clustering Algorithm. It keeps track of the evolution
        of the given parameters over all convergences.
        :param sample_count: count of the data sample at this iteration
        :param num_clusters: number of clusters at this iteration. = 0 if clusters have not been initialised for the
                             first time yet
        :param centroids: num_clusters * n_features ndarray of centroids at this iteration
        :param running_time: running time to convergence at this iteration. = nan if no convergence
                ie due to insufficient buffered data points
        :param num_iterations: number of iterations to convergence at this iteration. = nan if no convergence
        :param online_sse: sum of SSE over all clusters since initialise_clusters was last called
        :param online_compactness: mean of compactness over all clusters since initialise_clusters was last called
        :param online_dbi: mean of DBI over all clusters since initialise_clusters was last called
        :param masses: num_clusters * 1 ndarray of masses at this iteration
        :param counts: num_clusters * 1 ndarray of counts at this iteration
        :param centre_of_masses: num_clusters * 1 ndarray of centre_of_masses at this iteration
        :param probabilistic_counts: num_clusters * 1 ndarray of probabilistic_counts at this iteration
        :param alpha: significance threshold for setting optimal number of clusters. This will only change\
                      if the algorithm is allowed to update it. Commented out by default in
                      clustering_object.update_clusters
        :param gravitational_const: degree of gravitational pull new data points have on centre-of-masses. This will
                                    only change if the algorithm is allowed to update it. Commented out by default.
                                    in cluster_object.update_centre_of_mass
        """

        # append this iterations' values to lists
        self.sample_count.append(sample_count)
        self.num_clusters.append(num_clusters)
        self.alpha.append(alpha)
        self.gravitational_const.append(gravitational_const)
        self.centroids.append(centroids.tolist())
        self.running_time.append(running_time)
        self.num_iterations.append(num_iterations)
        self.online_sse.append(online_sse)
        self.online_compactness.append(online_compactness)
        self.online_dbi.append(online_dbi)
        self.masses.append(masses.tolist())
        self.counts.append(counts.tolist())
        self.probabilistic_counts.append(probabilistic_counts.tolist())
        self.centre_of_masses.append(centre_of_masses.tolist())

    def plot_parameter_evolution(self, data_step, customer_id, show):
        """
        plot lines giving number of clusters, count per cluster, mass per cluster with slider
        such that user can see evolution of these parameters over each sample count
        :param data_step: step of sample_count at which to plot clustering solution, computationally expensive for
                          smaller steps, but then a more gradual evolution will be visible
        :param show: boolean indicating if plot should be printed or not

        """

        # Create figure
        layout = go.Layout(title='Clustering',
                           xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=True),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=True),
                           hovermode='closest'
                           )

        fig = go.Figure(layout=layout)

        fig.add_trace(
            # optimal number of clusters at each data feed
            go.Scatter(x=np.arange(0, len(self.data), data_step),
                       y=self.num_clusters[::data_step],
                       name='No. Clusters',
                       showlegend=True
                       )
        )

        # fig.add_trace(
        #     # value of alpha at each data feed
        #     go.Scatter(x=np.arange(0, len(self.data), data_step),
        #             y=self.alpha[::data_step],
        #             name='Alpha',
        #             showlegend=True
        #             )
        # )
        #
        # fig.add_trace(
        #     # value of gravitational constant at each data feed
        #     go.Scatter(x=np.arange(0, len(self.data), data_step),
        #             y=self.gravitational_const[::data_step],
        #             name='Gravitational Constant',
        #             showlegend=True
        #             )
        # )

        if show:
            fig.show()

        try:
            fig.write_html(f"results/{self.test_name}/{customer_id}/parameter_evolution.html")
        except Exception as e:
            print(e)
        # throws error for large number of data points
        fig.write_image(f"results/{self.test_name}/{customer_id}/parameter_evolution_" + str(len(self.data)) + ".png")

    def plot_cluster_characteristic_evolution(self, data_step, customer_id, show, last_fig_only):
        """
        plot vertical grouped barchart giving number of clusters, count per cluster, mass per cluster with slider
        such that user can see evolution of these parameters over each sample count
        :param data_step: step of sample_count at which to plot clustering solution, computationally expensive for
                          smaller steps, but then a more gradual evolution will be visible
        :param last_fig_only: boolean indicating if only the last figure need be plotted
        :param show: boolean indicating if plot should be printed or not
        :param customer_id: id of customer data being plotted
        """

        num_traces_per_step = 3

        # check if the masses exist
        mass_exists = not np.any(np.isnan(self.masses[-1]))
        # if not exclude the masses to the traces
        if not mass_exists:
            num_traces_per_step -= 1

        # check if the probabilistic counts exist
        pc_exists = not np.any(np.isnan(self.probabilistic_counts[-1]))
        # if not exclude the probabilistic_counts to the traces
        if not pc_exists:
            num_traces_per_step -= 1

        # Create figure
        layout = go.Layout(title='Clustering',
                           xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=True),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=True),
                           hovermode='closest'
                           )

        fig = go.Figure(layout=layout)
        last_fig = go.Figure(layout=layout)

        evolution_range = [-1]
        if not last_fig_only:
            evolution_range = list(range(0, len(self.data), data_step)) + evolution_range

        # Add duo of traces for each slider step: mass and count for each cluster
        for idx in evolution_range:
            fig.add_trace(
                # cluster counts
                go.Bar(visible=False,
                       x=list(range(self.num_clusters[idx])),
                       y=self.counts[idx],
                       name='Counts',
                       showlegend=True
                       )
            )

            if mass_exists:
                fig.add_trace(
                    # cluster masses
                    go.Bar(visible=False,
                           x=list(range(self.num_clusters[idx])),
                           y=self.masses[idx],
                           name='Masses',
                           showlegend=True
                           )
                )

            if pc_exists:
                fig.add_trace(
                    # cluster probabilistic counts
                    go.Bar(visible=False,
                           x=list(range(self.num_clusters[idx])),
                           y=self.probabilistic_counts[idx],
                           name='Probabilistic Counts',
                           showlegend=True
                           )
                )

        # Make last traces visible and add last traces to last_fig object
        for trace_idx in range(len(fig.data) - num_traces_per_step, len(fig.data), 1):
            fig.data[trace_idx].visible = True
            last_fig.add_trace(fig.data[trace_idx])

        # Create and add slider. For each slider step, make corresponding convergence plot visible
        if not last_fig_only:
            steps = []
            for trace_idx, data_idx in zip(range(0, len(fig.data), num_traces_per_step), evolution_range):
                step = dict(
                    method="restyle",
                    args=["visible", [False] * 3 * len(fig.data)],
                    label=(f"No. Clusters = {self.num_clusters[data_idx]}."
                           f"Convergence Count = {self.sample_count[data_idx]:.0f}. "
                           f"No. Iterations to Convergence = {self.num_iterations[data_idx]:.0f}. "
                           f"Time to Convergence = {self.running_time[data_idx]:f}. "
                           f"SSE, Compactness, DBI since last Initialisation "
                           f"= {self.online_sse[data_idx]:.2f}, "
                           f"{self.online_compactness[data_idx]:.2f}, "
                           f"{self.online_dbi[data_idx]:.2f}")
                )
                # Toggle i'th traces to "visible"
                for inner_trace_idx in range(trace_idx, trace_idx + num_traces_per_step, 1):
                    step["args"][1][inner_trace_idx] = True

                steps.append(step)

            sliders = [dict(
                active=(len(fig.data) / num_traces_per_step) - 1,
                pad={"t": 50},
                steps=steps,
                font={'color': 'black'}
            )]

            fig.update_layout(
                barmode='group',
                sliders=sliders
            )

        if show:
            fig.show()

        try:
            fig.write_html(f"results/{self.test_name}/{customer_id}/characteristic_evolution_" + str(len(self.data)) + ".html")
        except Exception as e:
            print(e)
        # throws error for large number of data points
        last_fig.write_image(f"results/{self.test_name}/{customer_id}/characteristic_evolution_" + str(len(self.data)) + ".png")

    def plot_cluster_evolution(self, data_step, customer_id, show, last_fig_only):
        """
        plot centroids and centre of masses, and associated sse, count and mass
        at a given convergence count, with slider to show evolution.
        :param data_step: step of sample_count at which to plot clustering solution, computationally expensive for
                          smaller steps, but then a more gradual evolution will be visible
        :param show: boolean indicating if plot should be printed or not
        :param last_fig_only: boolean indicating if only the last figure need be plotted
        :param customer_id: id of customer data being plotted
        :return:
        """

        num_traces_per_step = 3

        # check if the centre-of-masses exist
        com_exists = not np.any(np.isnan(self.centre_of_masses[-1][-1]))
        # if not exclude the coms from the traces
        if not com_exists:
            num_traces_per_step -= 1

        # Create figure
        layout = go.Layout(title='Clustering',
                           xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=True),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=True),
                           hovermode='closest'
                           )

        fig = go.Figure(layout=layout)
        last_fig = go.Figure(layout=layout)

        evolution_range = [-1]
        if not last_fig_only:
            evolution_range = list(range(0, len(self.data), data_step)) + evolution_range

        # Add trio of traces for each slider step
        for idx in evolution_range:
            fig.add_trace(
                #  cluster centroids
                go.Scatter(visible=False,
                           x=[centroid[0] for centroid in self.centroids[idx]],
                           y=[centroid[1] for centroid in self.centroids[idx]],
                           name='Centroid',
                           mode='markers',
                           marker=go.scatter.Marker(symbol='x', size=20),
                           showlegend=False,
                           text=[(f"Cluster No. {c}."
                                  f"Mass = {self.masses[idx][c]}."
                                  f"Count = {self.counts[idx][c]}."
                                  f"Probabilistic Count = {self.probabilistic_counts[idx][c]}."
                                  ) for c in range(self.num_clusters[idx])]
                           )
            )

            if com_exists:
                fig.add_trace(
                    #  cluster centre-of-masses
                    go.Scatter(visible=False,
                               x=[com[0] for com in self.centre_of_masses[idx]],
                               y=[com[1] for com in self.centre_of_masses[idx]],
                               name='Centre Of Mass',
                               mode='markers',
                               marker=go.scatter.Marker(symbol='circle', size=20),
                               showlegend=False,
                               text=[(f"Cluster No. {c}."
                                      f"Mass = {self.masses[idx][c]}."
                                      f"Count = {self.counts[idx][c]}."
                                      f"Probabilistic Count = {self.probabilistic_counts[idx][c]}."
                                      ) for c in range(self.num_clusters[idx])]
                               )
                )

            # make bubble chart of invisible data stream, adding cluster information under color.
            fig.add_trace(
                # actual evolving dataset
                go.Scatter(
                    visible=False,
                    x=self.data['Hour'].values[0:idx],
                    y=self.data['Energy'].values[0:idx],
                    text=np.argmin(pairwise_distances(self.data.values,
                                                      np.vstack(self.centroids[-1])),
                                   axis=1),
                    name='Actual Data',
                    mode='markers',
                    showlegend=False
                )
            )

        # add last traces to last_fig object and
        # Make last traces visible initially
        for trace_idx in range(len(fig.data) - num_traces_per_step, len(fig.data), 1):
            fig.data[trace_idx].visible = True
            last_fig.add_trace(fig.data[trace_idx])

        if not last_fig_only:
            # Create and add slider. For each slider step, make that centroid, centre-of-mass, data trace
            # and convergence information visible
            steps = []
            for trace_idx, data_idx in zip(range(0, len(fig.data), num_traces_per_step), evolution_range):
                step = dict(
                    method="restyle",
                    args=["visible", [False] * 3 * len(fig.data)],
                    label=(f"No. Clusters = {self.num_clusters[data_idx]}."
                           f"Convergence Count = {self.sample_count[data_idx]:.0f}. "
                           f"No. Iterations to Convergence = {self.num_iterations[data_idx]:.0f}. "
                           f"Time to Convergence = {self.running_time[data_idx]:f}. "
                           f"SSE, Compactness, DBI since last Initialisation "
                           f"= {self.online_sse[data_idx]:.2f}, "
                           f"{self.online_compactness[data_idx]:.2f}, "
                           f"{self.online_dbi[data_idx]:.2f}")
                )
                # Toggle i'th traces to "visible"
                for inner_trace_idx in range(trace_idx, trace_idx + num_traces_per_step, 1):
                    step["args"][1][inner_trace_idx] = True

                steps.append(step)

            sliders = [dict(
                active=(len(fig.data) / num_traces_per_step) - 1,
                pad={"t": 50},
                steps=steps,
                font={'color': 'black'}
            )]

            fig.update_layout(
                sliders=sliders
            )

        if show:
            fig.show()
        try:
            fig.write_html(f"results/{self.test_name}/{customer_id}/cluster_evolution" + str(len(self.data)) + ".html")
        except Exception as e:
            print(e)
        # throws error for large number of data points
        # plotly.io.orca.ensure_server()
        # time.sleep(10)
        last_fig.write_image(f"results/{self.test_name}/{customer_id}/cluster_evolution_" + str(len(self.data)) + ".png")

    def finalise_at_index(self, customer_id, original_data, features):

        # update original dataset
        self.data = pd.DataFrame(
            original_data[[feat.name for feat in features]].loc[
                original_data.CustomerID == customer_id]).dropna().reset_index(drop=True)
        index = len(self.data) - 1

        # fetch the centroids at this index
        df = self.evolutionary_results.loc[
            (self.evolutionary_results.CustomerID == customer_id) &
            (self.evolutionary_results.ConvergenceCount == index), ['NumberClusters', 'Centroids', 'CentreOfMasses',
                                                                    'ProbabilisticCounts', 'Counts', 'Masses']]
        num_clusters = df.NumberClusters.iloc[0]
        results = pd.DataFrame()
        for col, items in df[['Centroids', 'CentreOfMasses']].iteritems():
            res = items.iloc[0].split(']')
            list_vals = []
            for c in range(num_clusters):
                regex = re.findall('([\d.e+-]+)', res[c])
                list_vals.append([float(g) for g in regex])
            results[col] = list_vals

        for col, items in df[['Counts', 'ProbabilisticCounts', 'Masses']].iteritems():
            res = items.iloc[0]
            regex = re.findall('([\d.e+-]+)', res)
            results[col] = [float(g) for g in regex]

        self.centroids = [results['Centroids'].values]
        self.centre_of_masses = [results['CentreOfMasses'].values]
        self.num_clusters = [num_clusters]
        self.masses = [results['Masses'].values]
        self.counts = [results['Counts'].values]
        self.probabilistic_counts = [results['ProbabilisticCounts'].values]

        # save metrics of each cluster at final_idx of data stream
        new_cluster_df = pd.DataFrame()
        new_cluster_df['Centroid'] = self.centroids[0]
        new_cluster_df['ClusterNumber'] = np.arange(num_clusters)

        # calculate actual final count, standard deviation, compactness, sse, dbi metrics
        std_dev, sse, count = self.calc_cluster_metrics(new_cluster_df)
        new_cluster_df['Standard Deviation'] = std_dev.tolist()
        new_cluster_df['SSE'] = sse
        new_cluster_df['Count'] = count
        new_cluster_df['Compactness'] = self.calc_compactness(new_cluster_df)
        new_cluster_df['DBI'] = self.calc_dbi(new_cluster_df)
        new_cluster_df['CustomerID'] = customer_id
        new_cluster_df['NumberConvergences'] = len(original_data) - 1

        self.clusters = self.clusters.append(new_cluster_df)
        self.clusters.to_csv(f'./results/{self.test_name}/clusters.csv', index=False)

        # save metrics of clustering solution at final_idx
        new_final_results_df = pd.DataFrame()
        new_final_results_df['NumberConvergences'] = [len(original_data) - 1]
        new_final_results_df['NumberClusters'] = [num_clusters]
        new_final_results_df['Compactness'] = [self.clusters.Compactness.mean()]
        new_final_results_df['DBI'] = [self.clusters.DBI.mean()]
        new_final_results_df['Separation'] = [self.calc_separation(new_cluster_df)]
        new_final_results_df['SSE'] = [self.clusters.SSE.sum()]
        new_final_results_df['ScenarioName'] = [self.test_name]
        new_final_results_df['CustomerID'] = [customer_id]

        self.final_results = self.final_results.append(new_final_results_df)
        self.final_results.to_csv(f'./results/final_results.csv', index=False)


    def finalise(self, customer_id, input_params, features, original_data, final_cluster_set, total_running_time):
        """
        updates dataframes and saves to csv files
        :param input_params: dictionary of input parameters
        :param original_data: full data set which was passed to clustering algorithm
        :param final_cluster_set: ClusterSet object with all clustering results at final_idx of data stream feed
        :param features: list of Feature objects which were passed to clustering algorithm
        """
        input_params['customer_ids'] = [input_params['customer_ids']]
        input_params['feature_names'] = [input_params['feature_names']]
        new_input_params_df = pd.DataFrame(input_params, index=[0])
        new_input_params_df.to_csv(f'./results/{self.test_name}/input_params.csv', index=False)

        # update original dataset
        self.data = pd.DataFrame(original_data[[feat.name for feat in features]]).dropna().reset_index(drop=True)

        # save metrics of each cluster at final_idx of data stream
        new_cluster_df = pd.DataFrame()
        new_cluster_df['Centroid'] = final_cluster_set.centroids.tolist()
        new_cluster_df['ClusterNumber'] = np.arange(final_cluster_set.num_clusters)

        # calculate actual final count, standard deviation, compactness, sse, dbi metrics
        std_dev, sse, count = self.calc_cluster_metrics(new_cluster_df)
        new_cluster_df['Standard Deviation'] = std_dev.tolist()
        new_cluster_df['SSE'] = sse
        new_cluster_df['Count'] = count
        new_cluster_df['Compactness'] = self.calc_compactness(new_cluster_df)
        new_cluster_df['DBI'] = self.calc_dbi(new_cluster_df)
        new_cluster_df['CustomerID'] = customer_id

        self.clusters = self.clusters.append(new_cluster_df)
        self.clusters.to_csv(f'./results/{self.test_name}/clusters.csv', index=False)

        # save metrics of each convergence over all data stream batches up to final_idx
        new_evolutionary_results_df = pd.DataFrame()
        new_evolutionary_results_df['ConvergenceCount'] = self.sample_count
        new_evolutionary_results_df['RunningTime'] = self.running_time
        new_evolutionary_results_df['NumberClusters'] = self.num_clusters
        new_evolutionary_results_df['NumberIterations'] = self.num_iterations
        new_evolutionary_results_df['SSESinceInit'] = \
            self.online_sse if self.online_sse is not None else [self.clusters.SSE.sum()]
        new_evolutionary_results_df['CompactnessSinceInit'] = \
            self.online_compactness if self.online_compactness is not None else [self.clusters.Compactness.mean()]
        new_evolutionary_results_df['DBISinceInit'] = \
            self.online_dbi if self.online_sse is not None else [self.clusters.DBI.mean()]
        new_evolutionary_results_df['CustomerID'] = customer_id
        new_evolutionary_results_df['Centroids'] = self.centroids
        new_evolutionary_results_df['CentreOfMasses'] = self.centre_of_masses
        new_evolutionary_results_df['ProbabilisticCounts'] = self.probabilistic_counts
        new_evolutionary_results_df['Counts'] = self.counts
        new_evolutionary_results_df['Masses'] = self.masses

        self.evolutionary_results = self.evolutionary_results.append(new_evolutionary_results_df)
        self.evolutionary_results.to_csv(f'./results/{self.test_name}/evolutionary_results.csv', index=False)

        # save metrics of clustering solution at final_idx
        new_final_results_df = pd.DataFrame()
        new_final_results_df['NumberConvergences'] = [len(original_data) - 1]
        new_final_results_df['NumberClusters'] = [final_cluster_set.num_clusters]
        new_final_results_df['Compactness'] = [self.clusters.Compactness.mean()]
        new_final_results_df['DBI'] = [self.clusters.DBI.mean()]
        new_final_results_df['Separation'] = [self.calc_separation(new_cluster_df)]
        new_final_results_df['SSE'] = [self.clusters.SSE.sum()]
        new_final_results_df['AverageNumberIterations'] = [self.evolutionary_results.NumberIterations.mean()]
        new_final_results_df['AverageRunningTime'] = [self.evolutionary_results.RunningTime.mean()]
        new_final_results_df['AbsoluteRunningTime'] = [total_running_time]
        new_final_results_df['ScenarioName'] = self.test_name
        new_final_results_df['CustomerID'] = customer_id

        self.final_results = self.final_results.append(new_final_results_df)

        # normalise relative running times
        self.final_results.RelativeRunningTime = self.final_results.AbsoluteRunningTime / \
                                                 self.final_results.AbsoluteRunningTime.min()

        self.final_results.to_csv(f'./results/final_results.csv', index=False)

    def calc_cluster_metrics(self, cluster_df):
        """
        calculate goodness-of-fit based on sse and standard deviation
        :param cluster_df: dataframe of clusters
        :return std_dev: standard deviation of each cluster
        :return sse: sum-squared-error of each cluster
        :return count: count of each cluster
        """
        n_clusters = len(cluster_df.index)
        centroids = np.vstack(cluster_df.Centroid.values)
        data = np.vstack(self.data.values)
        n_features = data.shape[1]
        cluster_distances = euclidean_distances(data, centroids, squared=True)
        closest_cluster_indices = np.argmin(cluster_distances, axis=1)

        unique_clusters, unique_count = np.unique(closest_cluster_indices, return_counts=True)
        count = np.zeros(n_clusters)
        for k in unique_clusters:
            count[k] = k

        # calculate sse of each cluster
        sse = np.zeros(len(cluster_df.index))

        for d, _ in self.data.iterrows():
            sse[closest_cluster_indices[d]] += cluster_distances[d, closest_cluster_indices[d]]

        # calculate standard deviation of each cluster
        std_dev = np.empty(shape=(n_clusters, n_features))
        for k in range(n_clusters):
            diff = data[closest_cluster_indices == k] - centroids[k]
            # calculate from covariance
            # product_sum = np.zeros((n_features, n_features))
            # calculate from variance
            product_sum = np.zeros(n_features)
            for d in range(len(diff)):
                # calculate from covariance
                # product_sum += diff[d, np.newaxis].T * diff[d, np.newaxis]
                # calculate from variance
                product_sum += diff[d] ** 2

            # calculate from covariance
            # std_dev = np.append(std_dev, np.sqrt((1 / (count[k] - 1)) * np.diag(product_sum)))
            # calculate from variance
            if count[k] > 1:
                std_dev[k] = np.sqrt((1 / (count[k] - 1)) * product_sum)

        return std_dev, sse, count

    def calc_compactness(self, cluster_df):
        """
        :param cluster_df: dataframe of clusters
        calculate compactness of fit
        :return compactness: compactness of each cluster
        """
        cluster_distances = euclidean_distances(self.data.values,
                                                np.vstack(cluster_df.Centroid), squared=True)
        closest_cluster_indices = np.argmin(cluster_distances, axis=1)

        compactness = [[] for k in cluster_df.index]

        for d, dp in self.data.iterrows():
            compactness[closest_cluster_indices[d]].append(
                norm(dp.values - np.vstack(cluster_df.loc[closest_cluster_indices[d], 'Centroid']), 1))

        for k, _ in cluster_df.iterrows():
            compactness[k] = np.mean(compactness[k])

        return compactness

    def calc_separation(self, cluster_df):
        """
        :param cluster_df: dataframe of clusters
        calculate separation of fit
        :return separation: separation of cluster set
        """
        separation = 0
        inter_centroid_distances = euclidean_distances(np.vstack(cluster_df.Centroid), squared=False)
        num_clusters = len(cluster_df.index)

        for k in range(num_clusters):
            for kk in range(k + 1, num_clusters):
                separation += inter_centroid_distances[k, kk]

        separation *= (2 / (num_clusters ** 2 - num_clusters))
        return separation

    def calc_dbi(self, cluster_df):
        """
        :param cluster_df: dataframe of clusters
        calculate Davies-Bouldin index of fit
        :return dbi: dbi of each cluster
        """
        dbi = 0
        inter_centroid_distances = euclidean_distances(np.vstack(cluster_df.Centroid), squared=False)
        num_clusters = len(cluster_df.index)

        for k, cluster1 in cluster_df.iterrows():
            dbi += np.max([(cluster1.Compactness + cluster2.Compactness) / inter_centroid_distances[k, kk]
                           for kk, cluster2 in cluster_df.iterrows() if kk != k])

        dbi *= 1 / num_clusters
        return dbi
