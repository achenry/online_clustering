import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
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

        self.batch_size = None

        # clusters.csv stores the precise centroid, standard deviation, l2_inertia, compactness and dbi values
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
                columns=['CustomerID', 'ClusterNumber', 'Centroid', 'L1_Inertia', 'Mass',
                         'Inertia', 'Compactness', 'DBI', 'NumberConvergences'])

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
                columns=['CustomerID', 'Compactness', 'DBI', 'Separation', 'Inertia',
                         'AverageRunningTimePerConvergence', 'NumberSamples', 'NumberClusters', 'AbsoluteRunningTime',
                         'RelativeRunningTime', 'ScenarioName', 'AverageRunningTimePerDataPoint'])

        # evolutionary_results.csv stores the evolving number of clusters, number of iterations, running time for, 
        # sse sum /compactness mean/dbi mean since last initialise_clusters was last called for each convergence
        # if an output evolutionary_results.csv file already exists, read it
        if os.path.exists(f'./results/{self.test_name}/evolutionary_results.csv'):
            self.evolutionary_results = pd.read_csv(f'./results/{self.test_name}/evolutionary_results.csv')
        # else, initialise a new dataframe
        else:
            self.evolutionary_results = pd.DataFrame(columns=['CustomerID', 'ConvergenceCount', 'NumberClusters',
                                                              'NumberIterations', 'RunningTime',
                                                              'Masses', 'inertias', 'Centroids'])

        # overall metrics for each iteration
        self.sample_count = []
        self.num_clusters = []
        self.centroids = []
        self.running_time = []
        self.num_iterations = []
        self.inertia = []
        self.compactness = []
        self.dbi = []
        self.alpha = []

        # lists of metrics for each cluster at each iteration
        self.masses = []
        self.inertias = []
        self.st_centroids = []

        self.data = None

    def reset(self):
        # overall metrics for each iteration
        self.sample_count = []
        self.num_clusters = []
        self.centroids = []
        self.running_time = []
        self.num_iterations = []
        self.inertia = []
        self.compactness = []
        self.dbi = []
        self.alpha = []

        # lists of metrics for each cluster at each iteration
        self.masses = []
        self.inertias = []
        self.st_centroids = []

    def update(self, sample_count, num_clusters, centroids, running_time, num_iterations, masses, inertia,
               compactness, dbi, inertias, st_centroids, alpha):
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
        :param masses: num_clusters * 1 ndarray of masses at this iteration
        :param st_centroids: num_clusters * 1 ndarray of st_centroids at this iteration
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
        self.inertia.append(inertia)
        self.compactness.append(compactness)
        self.dbi.append(dbi)
        self.running_time.append(running_time)
        self.num_iterations.append(num_iterations)

        self.centroids.append(centroids.tolist())
        self.masses.append(masses.tolist())
        self.inertias.append(inertias.tolist())
        self.st_centroids.append(st_centroids.tolist())

    def plot_parameter_evolution(self, data_step, customer_id, show):
        """
        plot lines giving number of clusters, count per cluster, mass per cluster with slider
        such that user can see evolution of these parameters over each sample count
        :param data_step: step of sample_count at which to plot clustering solution, computationally expensive for
                          smaller steps, but then a more gradual evolution will be visible
        :param show: boolean indicating if plot should be printed or not

        """

        # Create figure
        layout = go.Layout(xaxis={'showgrid': False,
                                  'zeroline': False,
                                  'showticklabels': True,
                                  'tick0': 0,
                                  'dtick': data_step,
                                  'title': 'Convergence Count'},
                           yaxis={'showgrid': False,
                                  'zeroline': False,
                                  'showticklabels': True,
                                  'title': 'No. Clusters'},
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

        if show:
            fig.show()

        try:
            fig.write_html(f"results/{self.test_name}/{customer_id}/params_evolution.html")
        except Exception as e:
            print(e)
        # throws error for large number of data points
        fig.write_image(f"results/{self.test_name}/{customer_id}/params_" + str(len(self.data)) + ".png")

    def plot_cluster_metric_evolution(self, data_step, customer_id, show, last_fig_only, export_all=False, actual_metrics=True):
        """
        plot vertical grouped barchart giving number of clusters, count per cluster, mass per cluster with slider
        such that user can see evolution of these parameters over each sample count
        :param data_step: step of sample_count at which to plot clustering solution, computationally expensive for
                          smaller steps, but then a more gradual evolution will be visible
        :param last_fig_only: boolean indicating if only the last figure need be plotted
        :param show: boolean indicating if plot should be printed or not
        :param customer_id: id of customer data being plotted
        """

        num_traces_per_step = 2

        # Create figure
        layout = go.Layout(xaxis={'showgrid': False,
                                  'zeroline': False,
                                  'showticklabels': True,
                                  'tickmode': 'linear',
                                  'tick0': 0,
                                  'dtick': 1,
                                  'title': 'Cluster No.'},
                           yaxis={'showgrid': False,
                                  'zeroline': False,
                                  'showticklabels': True,
                                  'title': 'Mass'},
                           yaxis2={'showgrid': False,
                                   'zeroline': False,
                                   'showticklabels': True,
                                   'overlaying': 'y',
                                   'side': 'right',
                                   'title': 'Inertia'},
                           hovermode='closest',
                           legend_orientation='h'
                           )

        fig = go.Figure(layout=layout)
        export_figs = []
        export_indices = []
        if last_fig_only:
            evolution_range = [len(self.num_clusters) - 1]
        else:
            evolution_range = list(range(0, len(self.num_clusters), data_step)) + [len(self.num_clusters) - 1]

        # Add duo of traces for each slider step: mass and count for each cluster
        for idx in evolution_range:

            if actual_metrics:
                cluster_df = pd.DataFrame()
                std_dev, inertia, count, mass = None, None, None, None
                if self.num_clusters[idx]:
                    cluster_df['Centroid'] = self.centroids[idx]
                    cluster_df['ClusterNumber'] = np.arange(self.num_clusters[idx])

                    # calculate actual final count, standard deviation, compactness, inertia, dbi metrics
                    std_dev, inertia, count, mass = self.calc_cluster_metrics(cluster_df)
                else:
                    cluster_df['Centroid'] = []
                    cluster_df['ClusterNumber'] = []

            fig.add_trace(
                # cluster masses
                go.Bar(visible=False,
                       x=np.arange(-0.2, self.num_clusters[idx], 1),
                       y=self.masses[idx] if not actual_metrics else mass,
                       width=0.4,
                       name='Mass',
                       showlegend=True
                       )
            )

            fig.add_trace(
                # cluster counts
                go.Bar(visible=False,
                       x=np.arange(0.2, self.num_clusters[idx], 1),
                       y=self.inertias[idx] if not actual_metrics else inertia,
                       name='Inertia',
                       width=0.4,
                       showlegend=True,
                       yaxis='y2'
                       )
            )

            if export_all or idx == evolution_range[-1]:
                export_figs.append(go.Figure(layout=layout))
                export_indices.append(idx)
                for idx in np.arange(num_traces_per_step)[::-1]:
                    export_figs[-1].add_trace(fig.data[-idx - 1])
                    export_figs[-1].data[num_traces_per_step - idx - 1].visible = True

            # add last traces to last_fig object and
            # Make last traces visible initially
            if idx == evolution_range[-1]:
                fig.data[idx].visible = True

        # Make last traces visible and add last traces to last_fig object
        # for trace_idx in range(len(fig.data) - num_traces_per_step, len(fig.data), 1):
        #     fig.data[trace_idx].visible = True
        #     last_fig.add_trace(fig.data[trace_idx])

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
                           f"Online Inertia, Compactness, DBI "
                           f"= {self.inertia[data_idx]:.2f}, "
                           f"{self.compactness[data_idx]:.2f}, "
                           f"{self.dbi[data_idx]:.2f}")
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
            fig.write_html(f"results/{self.test_name}/{customer_id}/metrics_evolution_" + str(len(self.data)) + ".html")
        except Exception as e:
            print(e)
        # throws error for large number of data points
        for export_idx, export_fig in zip(export_indices, export_figs):
            export_fig.write_image(f"results/{self.test_name}/{customer_id}/metrics_"
                                   + str(export_idx) + ".png")

    def plot_cluster_evolution(self, data_step, customer_id, show, last_fig_only, export_all=False):
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

        # Create figure
        layout = go.Layout(xaxis={'showgrid': False,
                                  'zeroline': False,
                                  'showticklabels': True,
                                   'title': 'Hour',
                                  'tickmode':'array',
                                  'tickvals':np.arange(1, 24,1),
        },
                           yaxis={'showgrid': False,
                                  'zeroline': False,
                                  'showticklabels': True,
                                  'title': 'Energy'},
                           hovermode='closest'
                           )

        fig = go.Figure(layout=layout)
        export_figs = []
        export_indices = []

        if last_fig_only:
            evolution_range = [len(self.num_clusters) - 1]
        else:
            evolution_range = list(range(0, len(self.num_clusters), data_step)) + [len(self.num_clusters) - 1]

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
                                  ) for c in range(self.num_clusters[idx])]
                           )
            )

            fig.add_trace(
                #  cluster centre-of-masses
                go.Scatter(visible=False,
                           x=[stc[0] for stc in self.st_centroids[idx]],
                           y=[stc[1] for stc in self.st_centroids[idx]],
                           name='Short-Term Centroid',
                           mode='markers',
                           marker=go.scatter.Marker(symbol='circle', size=20),
                           showlegend=False,
                           text=[(f"Cluster No. {c}."
                                  f"Mass = {self.masses[idx][c]}."
                                  ) for c in range(self.num_clusters[idx])]
                           )
            )

            # make bubble chart of invisible data stream, adding cluster information under color.
            fig.add_trace(
                # actual evolving dataset
                go.Scatter(
                    visible=False,
                    x=self.data['Hour'].values[0:(idx + 1) * self.batch_size],
                    y=self.data['Energy'].values[0:(idx + 1) * self.batch_size],
                    name='Actual Data',
                    mode='markers',
                    showlegend=False
                )
            )

            if export_all or idx == evolution_range[-1]:
                export_figs.append(go.Figure(layout=layout))
                export_indices.append(idx)
                for idx in np.arange(num_traces_per_step)[::-1]:
                    export_figs[-1].add_trace(fig.data[-idx - 1])
                    export_figs[-1].data[num_traces_per_step - idx - 1].visible = True

            # add last traces to last_fig object and
            # Make last traces visible initially
            if idx == evolution_range[-1]:
                fig.data[idx].visible = True

        # for trace_idx in range(len(fig.data) - num_traces_per_step, len(fig.data), 1):
        #     fig.data[trace_idx].visible = True
        #     export_figs.append(go.Figure(layout=layout))
        #     export_figs[-1].add_trace(fig.data[trace_idx])

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
                           f"Online Inertia, Compactness, DBI "
                           f"= {self.inertia[data_idx]:.2f}, "
                           f"{self.compactness[data_idx]:.2f}, "
                           f"{self.dbi[data_idx]:.2f}")
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
            fig.write_html(f"results/{self.test_name}/{customer_id}/centres_evolution" + str(len(self.data)) + ".html")
        except Exception as e:
            print(e)
        # throws error for large number of data points
        # plotly.io.orca.ensure_server()
        # time.sleep(10)
        for export_idx, export_fig in zip(export_indices, export_figs):
            export_fig.write_image(f"results/{self.test_name}/{customer_id}/centres_" + str(export_idx) + ".png")

    def finalise_at_index(self, customer_id, original_data, features):

        # update original dataset
        self.data = pd.DataFrame(
            original_data[[feat.name for feat in features]]).dropna().reset_index(drop=True)
        index = len(self.data) - 1

        # fetch the centroids at this index
        df = self.evolutionary_results.loc[
            (self.evolutionary_results.CustomerID == customer_id) &
            (self.evolutionary_results.ConvergenceCount == index), ['NumberClusters', 'Centroids', 'ShortTermCentroids',
                                                                    'Masses', 'Inertias']]
        num_clusters = df.NumberClusters.iloc[0]
        results = pd.DataFrame()
        for col, items in df[['Centroids', 'ShortTermCentroids']].iteritems():
            res = items.iloc[0].split(']')
            list_vals = []
            for c in range(num_clusters):
                regex = re.findall('([\d.e+-]+)', res[c])
                list_vals.append([float(g) for g in regex])
            results[col] = list_vals

        for col, items in df[['Masses', 'Inertias']].iteritems():
            res = items.iloc[0]
            regex = re.findall('([\d.e+-]+)', res)
            results[col] = [float(g) for g in regex]

        self.centroids = [results['Centroids'].values]
        self.st_centroids = [results['ShortTermCentroids'].values]
        self.num_clusters = [num_clusters]
        self.masses = [results['Masses'].values]
        self.inertias = [results['Inertias'].values]
        self.variances = [results['Variances'].values]

        # save metrics of each cluster at final_idx of data stream
        new_cluster_df = pd.DataFrame()
        new_cluster_df['Centroid'] = self.centroids[0]
        new_cluster_df['ClusterNumber'] = np.arange(num_clusters)

        # calculate actual final count, standard deviation, compactness, inertia, dbi metrics
        std_dev, inertia, count, mass = self.calc_cluster_metrics(new_cluster_df)
        new_cluster_df['Standard Deviation'] = std_dev.tolist()
        new_cluster_df['Inertia'] = inertia
        new_cluster_df['Count'] = count
        new_cluster_df['Mass'] = mass
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
        new_final_results_df['Inertia'] = [self.clusters.Inertia.sum()]
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
        input_params['customer_ids'] = customer_id
        input_params['feature_names'] = [input_params['feature_names']]
        self.batch_size = input_params['batch_size']
        new_input_params_df = pd.DataFrame(input_params, index=[0])
        self.input_params = new_input_params_df
        new_input_params_df = self.input_params.append(new_input_params_df)
        new_input_params_df.to_csv(f'./results/{self.test_name}/input_params.csv', index=False)

        # update original dataset
        self.data = pd.DataFrame(original_data[[feat.name for feat in features]]).dropna().reset_index(drop=True)

        # save metrics of each cluster at final_idx of data stream
        new_cluster_df = pd.DataFrame()
        new_cluster_df['Centroid'] = final_cluster_set.centroids.tolist()
        new_cluster_df['ClusterNumber'] = np.arange(final_cluster_set.num_clusters)

        # calculate actual final count, standard deviation, compactness, l2_inertia, dbi metrics
        std_dev, inertia, count, mass = self.calc_cluster_metrics(new_cluster_df)
        new_cluster_df['Standard Deviation'] = std_dev.tolist()
        new_cluster_df['Inertia'] = inertia
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
        new_evolutionary_results_df['CustomerID'] = customer_id
        new_evolutionary_results_df['Centroids'] = self.centroids
        new_evolutionary_results_df['ShortTermCentroids'] = self.st_centroids
        new_evolutionary_results_df['Masses'] = self.masses
        new_evolutionary_results_df['Inertia'] = self.inertias

        self.evolutionary_results = self.evolutionary_results.append(new_evolutionary_results_df)
        self.evolutionary_results.to_csv(f'./results/{self.test_name}/evolutionary_results.csv', index=False)

        # save metrics of clustering solution at final_idx
        new_final_results_df = pd.DataFrame()
        new_final_results_df['NumberSamples'] = [len(original_data)]
        new_final_results_df['NumberClusters'] = [final_cluster_set.num_clusters]
        new_final_results_df['Compactness'] = [self.clusters.Compactness.mean()]
        new_final_results_df['DBI'] = [self.clusters.DBI.mean()]
        new_final_results_df['Separation'] = [self.calc_separation(new_cluster_df)]
        new_final_results_df['Inertia'] = [self.clusters.Inertia.sum()]
        new_final_results_df['AverageNumberIterationsPerConvergence'] = [
            new_evolutionary_results_df.NumberIterations.mean()]
        new_final_results_df['AverageRunningTimePerConvergence'] = [new_evolutionary_results_df.RunningTime.mean()]
        if len(new_evolutionary_results_df.index) == 1:
            new_final_results_df['AverageNumberIterationsPerConvergence'] /= len(original_data)
            new_final_results_df['AverageRunningTimePerConvergence'] /= len(original_data)

        new_final_results_df['AbsoluteRunningTime'] = [total_running_time]
        new_final_results_df['AverageRunningTimePerDataPoint'] = [total_running_time / len(original_data)]
        new_final_results_df['ScenarioName'] = self.test_name
        new_final_results_df['CustomerID'] = customer_id

        self.final_results = self.final_results.append(new_final_results_df)

        # normalise relative running times
        self.final_results.RelativeRunningTime = self.final_results.AverageRunningTimePerConvergence / \
                                                 self.final_results.AverageRunningTimePerConvergence.min()

        self.final_results.to_csv(f'./results/final_results.csv', index=False)

    def calc_cluster_metrics(self, cluster_df):
        """
        calculate goodness-of-fit based on inertia and standard deviation
        :param cluster_df: dataframe of clusters
        :return std_dev: standard deviation of each cluster
        :return inertia: sum-squared-error of each cluster
        :return count: count of each cluster
        """
        fuzziness = self.input_params.loc[0, 'fuzziness']
        n_clusters = len(cluster_df.index)
        centroids = np.vstack(cluster_df.Centroid.values)
        data = np.vstack(self.data.values)
        n_samples = len(data)
        n_features = data.shape[1]
        cluster_distances = euclidean_distances(data, centroids, squared=True)
        closest_cluster_indices = np.argmin(cluster_distances, axis=1)

        unique_clusters, unique_count = np.unique(closest_cluster_indices, return_counts=True)
        count = np.zeros(n_clusters)
        for k in unique_clusters:
            count[k] = k

        # for hard clustering, where each row of fuzzy_labels (for each data point) will have a single nonzero value
        # of 1 corresponding to the cluster to which that data point certainly belongs
        if fuzziness == 1:
            D = np.zeros((n_samples, n_clusters))
            for data_point_idx, closest_cluster_idx in enumerate(closest_cluster_indices):
                D[data_point_idx, closest_cluster_idx] = 1
        # for fuzzy clustering, the values over each row of fuzzy_labels will sum to 1
        else:
            with np.errstate(divide='ignore'):
                D = 1.0 / cluster_distances

            x_eq_centroid_indices = np.argwhere(D == np.inf)
            D[x_eq_centroid_indices[:, 0], :] = 0
            D[x_eq_centroid_indices[:, 0], x_eq_centroid_indices[:, 1]] = 1

            D **= np.divide(1.0, (fuzziness - 1))
            D /= np.sum(D, axis=1)[:, np.newaxis]

        # n_samples * n_clusters ndarray of probability of membership of each sample to each cluster
        fuzzy_labels = D
        mass = np.sum(fuzzy_labels, axis=0)

        # calculate squared error of each cluster
        inertia = np.zeros(len(cluster_df.index))

        for d, _ in self.data.iterrows():
            inertia[closest_cluster_indices[d]] += cluster_distances[d, closest_cluster_indices[d]]

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

        return std_dev, inertia, count, mass

    def calc_compactness(self, cluster_df):
        """
        :param cluster_df: dataframe of clusters
        calculate compactness of fit
        :return compactness: compactness of each cluster
        """
        cluster_distances = euclidean_distances(self.data.values,
                                                np.vstack(cluster_df.Centroid), squared=False)
        closest_cluster_indices = np.argmin(cluster_distances, axis=1)
        num_clusters = len(cluster_df.Centroid.index)

        compactness = np.zeros(len(cluster_df.index))

        for d, dp in self.data.iterrows():
            compactness[closest_cluster_indices[d]] += \
                norm(dp.values - np.vstack(cluster_df.loc[closest_cluster_indices[d], 'Centroid']), 1)

        compactness /= np.sum(closest_cluster_indices == np.arange(num_clusters)[:, np.newaxis], axis=1)

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
