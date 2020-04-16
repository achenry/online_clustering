import plotly.graph_objs as go
import plotly.express as ex

class ClusteringPlot:

    def __init__(self, centroids, original_data):
        self.centroids = centroids
        self.data = original_data

    def plot(self):
        pass