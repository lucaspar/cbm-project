import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA

DIR_PLOTS = os.path.join('..', "outputs", "plots")

def add_value_labels(ax, spacing=5, fontsize=8):
    """Add labels to the end of each bar in a bar chart.
    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """
    # place a label for each bar
    for rect in ax.patches:
        # get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        # vertical alignment for positive values
        va = 'bottom'
        # if value of bar is negative: Place label below bar
        if y_value < 0:
            spacing *= -1     # invert space to place label below
            va = 'top'      # vertically align label at top
        # use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)
        # Create annotation
        ax.annotate(
            text=label,
            xy=(x_value, y_value),
            fontsize=fontsize,
            xytext=(0, spacing),        # vertically shift label by `space`
            textcoords="offset points", # interpret `xytext` as offset in points
            ha='center',                # horizontally center label
            va=va                       # vertically align label differently for
                                        # positive and negative values.
        )

def find_elbow(embedding, knee_id="", n_neighbors=4):
    """
        Given an embedding, finds and plots the nearest-neighbor distances in that space.
        It returns the coordinates of the 'knee' of this curve, point which indicates the
        optimal 'maximum distance' for a clustering algorithm such as DBSCAN to determine outliers.
    """

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn_fit = knn.fit(embedding)
    distances, _ = knn_fit.kneighbors(embedding)
    distances = np.sort(distances, axis=0)[:,1]
    kneedle = KneeLocator(np.linspace(0, len(distances)-1, len(distances)), distances, S=1.0, curve="convex", direction="increasing")
    knee_x, knee_y = kneedle.elbow, distances[int(kneedle.elbow)]

    print('Knee: the Maximum Curvature point is (x = {}, y = {})'.format(knee_x, knee_y))

    kneedle.plot_knee()
    PLT_SUBPLOT = "knee"
    os.makedirs(os.path.join(DIR_PLOTS, PLT_SUBPLOT), exist_ok=True)
    knee_fname = os.path.join(DIR_PLOTS, PLT_SUBPLOT, "{}.pdf".format(knee_id))
    plt.savefig(knee_fname)
    plt.close()
    print("Saved knee plot as '{}'".format(knee_fname))

    return knee_x, knee_y

def plot_scatter(embedding_df, clustering_df):
    COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_map, cluster_freqs = dict(), dict()
    frequencies_iter = clustering_df['cluster'].value_counts().iteritems()
    for idx, (cluster, freq) in enumerate(frequencies_iter):
        color = COLORS[idx % len(COLORS)]
        color_map[cluster] = color
        cluster_freqs[cluster] = freq
    x = cluster_freqs.keys()
    heights = cluster_freqs.values()

    fig, ax = plt.subplots()
    marker_size = 5 / np.log10(embedding_df["emb_x"].shape[0]) + 4
    ax = pca_df.plot.scatter("emb_x", "emb_y",
        ax=ax,
        color=clustering_df['cluster'].map(color_map),
        s=marker_size,
        # cmap='viridis',
    )
    if -1 in color_map:
        outlier_color = color_map[-1] if -1 in color_map else None
        outlier_patch = mpatches.Patch(color=outlier_color, label='DBSCAN Outliers')
        ax.legend(handles=[outlier_patch])
    fig.savefig('scatter.pdf')

    fig, ax = plt.subplots()
    ax.bar(x, height=heights, color=list(color_map.values()))
    add_value_labels(ax, fontsize=6)
    fig.savefig('bars.pdf')
    return

embeddings = {}
with open('lucas_embeddings.txt') as infile:
    for line in infile:
        vertex, vector = line.strip().split('\t')
        embeddings[int(vertex)] = list(map(float, vector.replace(',', '').replace('[', '').replace(']', '').split(' ')))

X = np.asarray([vector for vector in embeddings.values()])

x, y = find_elbow(X)

clustering = DBSCAN(eps=y, min_samples=2).fit(X)
clustering_dict = {'vertex': list(range(len(clustering.labels_))), 'cluster': clustering.labels_}
clustering_df = pd.DataFrame(clustering_dict)

pca = PCA(n_components=2)
Y = pca.fit_transform(X)
embedding_df = pd.DataFrame(X)
pca_df = pd.DataFrame(Y)
pca_df.columns = ['emb_x', 'emb_y']

plot_scatter(pca_df, clustering_df)
