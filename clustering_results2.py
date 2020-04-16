import plotly.graph_objs as go
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
import numpy as np
from numpy.linalg import norm
import os


class ClusteringResults:

    def __init__(self, original_data, features):

        if os.path.exists('clusters.csv'):
            self.clusters = pd.read_csv('clusters.csv')
        else:
            self.clusters = pd.DataFrame(columns=['Test', 'Centroid', 'CentreOfMass', 'Covariance', 'Count', 'SSE',
                                                  'Mass', 'Compactness', 'DBI'])

        if os.path.exists('final_results.csv'):
            self.final_results = pd.read_csv('final_results.csv')
        else:
            self.final_results = pd.DataFrame(
                columns=['Test', 'Compactness', 'DBI', 'Separation', 'SSE', 'AverageRunningTime',
                         'AverageNumberIterations', 'NumberConvergences',
                         'NumberClusters'])

        if os.path.exists('evolutionary_results.csv'):
            self.evolutionary_results = pd.read_csv('evolutionary_results.csv')
        else:
            self.evolutionary_results = pd.DataFrame(columns=['Test', 'ConvergenceCount', 'NumberClusters',
                                                              'NumberIterations', 'RunningTime', 'OnlineSSE'])

        self.convergence_count = []
        self.num_clusters = []
        self.centroids = []
        self.running_time = []
        self.num_iterations = []
        self.online_sses = []
        self.masses = []
        self.counts = []
        self.centre_of_masses = []

        # average time required for new data point(s) convergence
        # self.results.AverageRunningTime = timedelta(seconds=0)
        # # average number of iterations per new data point(s) convergence
        # self.results.AverageNumberIterations = 0
        # self.results.NumberConvergences = 0

        # clustered_data_df.Centroids.unique().values

        self.data = pd.DataFrame(original_data[[feat.name for feat in features]]).dropna().reset_index(drop=True)

    def update(self, convergence_count, num_clusters, centroids, running_time, num_iterations, online_sses, masses,
               counts, centre_of_masses):
        self.convergence_count.append(convergence_count)
        self.num_clusters.append(num_clusters)
        self.centroids.append(centroids)
        self.running_time.append(running_time)
        self.num_iterations.append(num_iterations)
        self.online_sses.append(online_sses)
        self.masses.append(masses)
        self.counts.append(counts)
        self.centre_of_masses.append(centre_of_masses)

    def plot_num_clusters_evolution(self):
        """
        plot horizontal lines representing centroids, which connect on merge and disconnect on diverge
        """
        pass

    def plot_cluster_characteristic_evolution(self):
        """
        plot vertical grouped barchart giving number of clusters, count per cluster, mass per cluster
        """
        pass

    def plot_cluster_evolution(self):
        """
        plot centroids and centre of masses,
        and associated sse, count and mass
        at a given convergence count, with slider to show evolution.
        """

        # Create figure
        layout = go.Layout(title='Clustering',
                           xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=False),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=False),
                           hovermode='closest'
                           )
        fig = go.Figure(layout=layout)

        # Add traces, one for each slider step
        for idx in range(len(self.centroids)):
            fig.add_trace(
                #  cluster centroids
                go.Scatter(visible=False,
                           x=[centroid[0] for centroid in self.centroids[idx]],
                           y=[centroid[1] for centroid in self.centroids[idx]],
                           name='Centroid',
                           mode='markers',
                           marker=go.scatter.Marker(symbol='x', size=20),
                           showlegend=False,
                           text='Cluster SSE Values = %s' % (str(self.online_sses[idx])),
                           )
            )

            fig.add_trace(
                #  cluster centre-of-masses
                go.Scatter(visible=False,
                           x=[com[0] for com in self.centre_of_masses[idx]],
                           y=[com[1] for com in self.centre_of_masses[idx]],
                           name='Centre Of Mass',
                           mode='markers',
                           marker=go.scatter.Marker(symbol='circle', size=20),
                           showlegend=False,
                           text='Cluster Masses = %s\n'
                                'Cluster Counts = %s' % (str(self.masses[idx]),
                                                         str(self.counts[idx])),
                           )
            )

        # Make last trace visible
        fig.data[-1].visible = True

        # make bubble chart of invisible data stream, adding cluster information under color.
        fig.add_trace(go.Scatter(
            x=self.data['Hour'].values,
            y=self.data['Energy'].values,
            visible=False,
            text=np.argmin(pairwise_distances(self.data.values,
                                              np.vstack(self.centroids[-1])),
                           axis=1),
            name='Actual Data',
            mode='markers',
            showlegend=False))

        # Create and add slider
        steps = []
        for idx in range(len(self.convergence_count)):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig.data)],
                label=["Convergence Count = %.0f\n"
                       "Number of Iterations to Convergence = %.0f\n"
                       "Processing Time to Convergence = %f\n,"
                       "Online SSE = %.2f" % (self.convergence_count[idx], self.num_iterations[idx],
                                            self.running_time[idx], sum(self.online_sses[idx]))]
            )
            # Toggle i'th trace to "visible"
            step["args"][1][idx] = True
            steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Convergence Count: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.57,
                    y=1.2,
                    buttons=list([
                        dict(
                            label="Toggle Actual Data Plot",
                            method="update",
                            args=["visible",
                                  [True if fig.data[idx].visible else False for idx in range(len(fig.data) - 1)]
                                  + [True if not fig.data[-1].visible else False]]
                        ),
                    ]),
                )
            ]
        )

        fig.show()

    def plot_clusters(self):

        # make bubble chart of data stream, adding cluster information under color.
        data_trace = go.Scatter(x=self.data['Energy'].values,
                                y=np.zeros(len(self.data.index)),  # self.data['Hour'].values,#
                                # y=self.data['Energy'],
                                text=np.argmin(pairwise_distances(self.data.values,
                                                                  np.vstack(self.centroids[-1])),
                                               axis=1),
                                name='',
                                mode='markers',
                                showlegend=False
                                )

        # Represent cluster centers.
        cluster_trace = go.Scatter(x=[row[0] for row in self.centroids[-1]],
                                   y=np.zeros(len(self.data.index)),  # [row[0] for row in self.clusters['Centroid']],#
                                   name='',
                                   mode='markers',
                                   marker=go.scatter.Marker(symbol='x', size=20),
                                   # color=range(len(self.centroids))),
                                   showlegend=False
                                   )

        layout = go.Layout(title='Clustering',
                           xaxis=go.layout.XAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=False),
                           yaxis=go.layout.YAxis(showgrid=False,
                                                 zeroline=False,
                                                 showticklabels=False),
                           hovermode='closest'
                           )

        # data = go.Data([data_trace, cluster_trace])
        fig = go.Figure(data=[data_trace, cluster_trace], layout=layout)
        fig.show()

    # def update(self, num_iters, running_time):
    #     pass
    # self.final_results.NumberConvergences += 1
    # self.final_results.AverageNumberIterations += (num_iters - self.final_results.AverageNumberIterations) \
    #                                         / self.final_results.NumberConvergences
    # self.final_results.AverageRunningTime += (running_time - self.final_results.AverageRunningTime) \
    #                                    / self.final_results.NumberConvergences

    def finalise(self, test_name, final_cluster_set):

        new_cluster_df = pd.DataFrame()
        new_cluster_df['Test'] = test_name
        new_cluster_df['Centroid'] = final_cluster_set.centroids.tolist()
        new_cluster_df['CentreOfMass'] = final_cluster_set.centre_of_masses.tolist()
        new_cluster_df['Covariance'] = final_cluster_set.covariances.tolist()
        new_cluster_df['Count'] = final_cluster_set.counts
        new_cluster_df['Mass'] = final_cluster_set.masses

        new_cluster_df['Compactness'] = self.calc_cp(new_cluster_df)
        new_cluster_df['SSE'] = self.calc_sse(new_cluster_df)
        new_cluster_df['DBI'] = self.calc_dbi(new_cluster_df)

        self.clusters = self.clusters.append(new_cluster_df)
        self.clusters.to_csv('clusters.csv')

        new_evolutionary_results_df = pd.DataFrame()

        new_evolutionary_results_df['Test'] = test_name
        new_evolutionary_results_df['ConvergenceCount'] = self.convergence_count
        new_evolutionary_results_df['RunningTime'] = self.running_time
        new_evolutionary_results_df['NumberClusters'] = self.num_clusters
        new_evolutionary_results_df['NumberIterations'] = self.num_iterations
        new_evolutionary_results_df['OnlineSSE'] = self.online_sses

        self.evolutionary_results = self.evolutionary_results.append(new_evolutionary_results_df)
        self.evolutionary_results.to_csv('evolutionary_results.csv')

        new_final_results_df = pd.DataFrame()

        new_final_results_df['Test'] = test_name
        new_final_results_df['NumberConvergences'] = self.convergence_count[-1]
        new_final_results_df['NumberClusters'] = self.num_clusters[-1]
        new_final_results_df['Compactness'] = self.clusters.Compactness.mean()
        new_final_results_df['DBI'] = self.clusters.DBI.mean()
        new_final_results_df['Separation'] = self.calc_sp(new_cluster_df)
        new_final_results_df['SSE'] = self.clusters.SSE.sum()
        new_final_results_df['AverageNumberIterations'] = self.evolutionary_results.NumberIterations.mean()
        new_final_results_df['AverageRunningTime'] = self.evolutionary_results.RunningTime.mean()

        self.final_results = self.final_results.append(new_final_results_df)
        self.final_results.to_csv('final_results.csv')

    def calc_sse(self, cluster_df):
        """
        calculate goodness-of-fit based on sse
        :return:
        """
        cluster_distances = euclidean_distances(self.data.values, np.vstack(cluster_df.Centroid),
                                                squared=True)
        closest_cluster_indices = np.argmin(cluster_distances, axis=1)

        sse = np.zeros(len(cluster_df.index))

        for d, _ in self.data.iterrows():
            sse[closest_cluster_indices[d]] += cluster_distances[d, closest_cluster_indices[d]]

        return sse

    def calc_cp(self, cluster_df):
        """
        calculate compactness of fit
        """
        cluster_distances = euclidean_distances(self.data.values,
                                                np.vstack(cluster_df.Centroid), squared=True)
        closest_cluster_indices = np.argmin(cluster_distances, axis=1)

        cp = [[] for k in cluster_df.index]

        for d, dp in self.data.iterrows():
            cp[closest_cluster_indices[d]].append(
                norm(dp.values - cluster_df.loc[closest_cluster_indices[d], 'Centroid'], 1))

        for k, _ in cluster_df.iterrows():
            cp[k] = np.mean(cp[k])

        return cp

    def calc_sp(self, cluster_df):
        """
        caculate separation of fit
        """
        num_clusters = len(cluster_df.index)
        sp = (2 / (num_clusters ** 2 - num_clusters)) * \
             np.sum(np.tril(euclidean_distances(np.vstack(cluster_df.Centroid), squared=False)))
        return sp

    def calc_dbi(self, cluster_df):
        """
        calculate Davies-Bouldin index of fit
        """
        dbi = []

        for k, cluster1 in cluster_df.iterrows():
            dbi.append(np.max([(cluster1.Compactness + cluster2.Compactness)
                             / norm(np.array(cluster1.Centroid) - np.array(cluster2.Centroid), 2)
                             for kk, cluster2 in cluster_df.iterrows() if kk != k]))

        return dbi
