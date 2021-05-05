#!/bin/env python
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering


warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

DIR_DATASETS_BASE = "/media/Data/datasets/graphs/"
DIR_ENRON_DB = os.path.join(DIR_DATASETS_BASE, "enron/preprocessed")
DIR_PLOTS = os.path.join("outputs", "plots")
DIR_EMBEDDINGS = os.path.join("outputs", "embeddings")
DIR_PREPROCESSED = os.path.join(DIR_EMBEDDINGS, "preprocessed")
DIR_CACHE = "./cache"
SUBSET_SIZE = -10_000
DB_MAX_CLUSTERS = 32
RANDOM_STATE = 42
COLORS = list(mcolors.TABLEAU_COLORS.values())
# COLORS = list(mcolors.CSS4_COLORS.values())

for d in [DIR_CACHE, DIR_PLOTS, DIR_PREPROCESSED]:
    os.makedirs(d, exist_ok=True)


def load_enron_as_graph(db_path):
    """Loads pre-processed Enron dataset as a graph."""

    # load dataset as a dataframe
    print("Loading {}...".format(db_path))
    email_df = pd.read_csv(db_path)
    subset = email_df[0:SUBSET_SIZE] if SUBSET_SIZE > 0 else email_df

    # remove spaces from emails
    for col in ["From", "To"]:
        email_df[col] = email_df[col].apply(lambda x: x.replace(" ", ""))

    # build graph
    graph = nx.from_pandas_edgelist(subset, 'From', 'To', edge_attr=['Dates'])

    return graph


def run_node2vec(graph, emb_out_file='embedding.emb'):
    """Runs Node2Vec embedding on a graph, outputs to disk as `emb_file`."""

    node2vec = Node2Vec(graph, dimensions=2, walk_length=20, num_walks=10, workers=4)

    # learn embeddings
    model = node2vec.fit(window=10, min_count=1)
    # model.wv.most_similar('1')

    # save outputs
    model.wv.save_word2vec_format(emb_out_file)

    return model


def norm_embeddings(embedding_df, fname='embedding_norm.emb', write_to_csv=True):
    """Converts email addresses in embedding to numeric IDs."""

    embedding_df = embedding_df.copy()
    embedding_df = embedding_df.reset_index()
    embedding_df.insert(0, 'node_id', range(0, len(embedding_df)))
    embedding_df.rename(
        inplace=True,
        columns={
            "index": "node_name",
            embedding_df.columns[-2]: "emb_x",      # second to last
            embedding_df.columns[-1]: "emb_y"       # last
        },
    )
    selected_cols = ["node_id", "emb_x", "emb_y"]

    if write_to_csv:
        embedding_df[selected_cols].to_csv(fname, index=False, header=False, sep=' ')

    return embedding_df


def plot_everything(graph=None, embedding_df=None, clusters=None, meta="", plot_graph=True, plot_scatter=True, cluster_col="cluster"):
    """Create plots given a clusters and node_ids, and/or a graph."""

    assert [x in meta for x in ["embedding", "clustering"]]
    embedding_df = embedding_df.copy()
    embedding_df = embedding_df.set_index("node_name")

    unique_clusters = sorted(embedding_df[cluster_col].unique())
    color_map = { col: COLORS[idx % len(COLORS)] for idx, col in enumerate(unique_clusters) }

    # scatter plot of nodes by cluster
    if plot_scatter and all(x is not None for x in [embedding_df]):

        PLT_SUBDIR = "scatter"
        os.makedirs(os.path.join(DIR_PLOTS, PLT_SUBDIR), exist_ok=True)
        fname = os.path.join(DIR_PLOTS, PLT_SUBDIR, "{}.pdf".format(cluster_col))

        ax = embedding_df.plot.scatter("emb_x", "emb_y",
            color=embedding_df[cluster_col].map(color_map), s=5 / np.log10(embedding_df["emb_x"].shape[0]) + 4, cmap='viridis'
        )
        ax.set_title("{} {}".format(meta["embedding"], meta["clustering"]))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print("Saved scatter plot as '{}'".format(fname))

    # graph plot color-coded by cluster
    if plot_graph and all(x is not None for x in [graph, embedding_df]):

        PLT_SUBDIR = "graph"
        os.makedirs(os.path.join(DIR_PLOTS, PLT_SUBDIR), exist_ok=True)

        # params
        all_layouts = {
            # "spectral_layout": {},    # no convergence after 28k iterations
            # "spiral_layout": {},
            "spring_layout": {
                "k": 0.1,
            },
        }

        # get list of node colors with the order they appear in the graph
        node_colors = list()
        for node_name in graph:
            try:
                node_color = embedding_df.loc[node_name][cluster_col]
                node_colors.append(node_color)
                # print("OK: {}".format(node_name))
            except Exception as err:
                print("Warning:\t{}".format(err))
                node_colors.append(666)

        for layout_style, layout_kwargs in all_layouts.items():

            fname = os.path.join(DIR_PLOTS, PLT_SUBDIR, "{}_{}.pdf".format(cluster_col, layout_style))

            plt.figure(figsize=(20,20))
            # pos = nx.spring_layout(graph)
            pos = getattr(nx, layout_style)(graph, **layout_kwargs)
            nx.draw_networkx(
                graph,
                pos,
                node_size=25,
                node_color=node_colors,
                cmap='viridis',
                with_labels=False,
                edge_color=(0, 0, 1, 0.05),
                width=2,
            )
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print("Saved graph plot as '{}'".format(fname))


def db_index_search(emb, min_clusters=2, max_clusters=8, plot_scores=False, title_suffix="", plot_id=None):
    """
        Performs a search for the optimal number of clusters in an embedded space using Davies-Bouldin index.
        Returns:
            best_k:         optimal number of clusters by computing Davies-Bouldin index.
            best_db_index:  optimal Davies-Bouldin index achieved by `best_k`.
            best_kmeans:    the optimal clustering.
            scores:         all DB indices computed as a dictionary { "k": "DB index" }.
    """

    PLT_SUBDIR = "db-index"
    os.makedirs(os.path.join(DIR_PLOTS, PLT_SUBDIR), exist_ok=True)

    best_k = min_clusters
    best_db_index = np.inf
    scores = dict()
    for current_k in range(min_clusters, max_clusters + 1, 1):
        kmeans = KMeans(n_clusters=current_k, random_state=RANDOM_STATE).fit(emb)
        current_db_index = davies_bouldin_score(emb, kmeans.labels_)
        scores[current_k] = current_db_index
        if current_db_index < best_db_index:
            best_k, best_db_index, best_kmeans = current_k, current_db_index, kmeans

    if plot_scores:
        x, y = list(scores.keys()), list(scores.values())
        plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
        plt.title("DB index scores{}".format(title_suffix))
        if plot_id is None:
            plot_id = str(datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S"))
        fname = os.path.join(DIR_PLOTS, PLT_SUBDIR, "{}.pdf".format(plot_id))
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print("Saved Davies-Bouldin plot as '{}'".format(fname))

    return {
        "best_k":           best_k,
        "best_db_index":    best_db_index,
        "best_kmeans":      best_kmeans,
        "scores":           scores,
    }


def plot_db_scores():
    """Plots DB score curves in the same plot."""

    PLT_SUBDIR = "db-index"
    os.makedirs(os.path.join(DIR_PLOTS, PLT_SUBDIR), exist_ok=True)
    fname = os.path.join(DIR_PLOTS, PLT_SUBDIR, "all-scores.pdf")

    data = {
        "n2v_db_search": {
            "label": "Node2Vec",
        },
        "sc_db_search": {
            "label": "Spectral Clustering",
        },
        "dw_db_search": {
            "label": "DeepWalk",
        },
    }
    for fname in data.keys():
        fpath = os.path.join(DIR_CACHE, fname)
        cached = joblib.load(fpath)
        data[fname].update(cached)

    for fname, d in data.items():
        scores = d["scores"]
        label = d["label"]
        x, y = list(scores.keys()), list(scores.values())
        plt.plot(x, y, marker='o', linestyle='dashed', label=label, linewidth=2, markersize=5 / np.log10(len(x)) + 4)

    plt.xlabel('Number of clusters (k)')
    plt.ylabel('DB score')
    plt.title("Davies-Bouldin clustering scores (lower is better)".format())
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print("Saved joint Davies-Bouldin plot as '{}'".format(fname))


def sc_cluster(data_graph, clustering=["kmeans"], plot_db_index=True):
    """
        Runs spectral embedding + clustering on a graph dataset,
        optionally computing the DB-index for clustering evaluation.
    """

    clustering = [ clustering ] if isinstance(clustering, str) else clustering

    adj_mat = nx.to_numpy_matrix(data_graph)
    node_list = list(data_graph.nodes())
    unique_list = list(set(node_list))

    # map emails to numeric ids and vice-versa
    emails_to_node_ids, node_ids_to_emails = dict(), dict()
    for node_id, email in enumerate(sorted(unique_list)):
        node_ids_to_emails[node_id] = email
        emails_to_node_ids[email] = node_id

    # get embeddings of graph
    spemb = SpectralEmbedding(
        n_components=2,
        affinity='precomputed',
        random_state=RANDOM_STATE,
    )
    sc_emb = spemb.fit_transform(adj_mat)

    # create spectral embedding dataframe
    rows = list()
    for node_id, (emb_x, emb_y) in enumerate(sc_emb):
        node_name = node_ids_to_emails[node_id]
        row = [node_id, node_name, emb_x, emb_y]
        rows.append(row)
    sc_df = pd.DataFrame(data=rows, columns=['node_id', 'node_name', 'emb_x', 'emb_y'])

    # run k-means on spectral embeddings
    if "kmeans" in clustering:

        if plot_db_index:
            db_search = db_index_search(
                emb=sc_emb,
                min_clusters=2,
                max_clusters=DB_MAX_CLUSTERS,
                plot_scores=True,
                title_suffix=" - Spectral Clustering",
                plot_id="sc",
            )
            sc_kmeans = db_search["best_kmeans"]
            joblib.dump(db_search, os.path.join(DIR_CACHE, "sc_db_search"), compress=4)
        else:
            sc_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(sc_emb)

        sc_df.insert(
            loc=len(sc_df.columns),
            column='sc_kmeans',
            value=sc_kmeans.labels_,
            allow_duplicates=True,
        )

    if "dbscan" in clustering:

        # find optimal number of neighbors in cluster
        method_id = "sc"
        _, epsilon = find_elbow(sc_emb, knee_id=method_id)
        print("Elbow: {}".format(epsilon))

        dbscan = DBSCAN(eps=epsilon, metric="euclidean", min_samples=5).fit(sc_emb)
        davies_bouldin_index = davies_bouldin_score(sc_emb, dbscan.labels_)
        print("SC DB Index: {}".format(davies_bouldin_index))

        # augment dataframe with the clusters found
        sc_df.insert(
            loc=len(sc_df.columns),
            column='sc_dbscan',
            value=dbscan.labels_,
            allow_duplicates=True,
        )

    print(sc_df.head(10))
    return sc_df


def n2v_cluster(data_graph, clustering=["kmeans"], plot_db_index=True):
    """Runs node2vec embedding + clustering on dataset."""

    clustering = [ clustering ] if isinstance(clustering, str) else clustering

    # init
    using_subset_size = "full" if SUBSET_SIZE <= 0 else SUBSET_SIZE
    n2v_emb_file = os.path.join(DIR_EMBEDDINGS, 'n2v_embedding_{}.emb'.format(str(using_subset_size)))
    n2v_emb_file_norm = "n2v_norm.emb"

    # load graph and get embeddings
    if not os.path.exists(n2v_emb_file):
        _ = run_node2vec(graph=data_graph, emb_out_file=n2v_emb_file)

    # reload embeddings and convert their email addresses to numeric IDs
    n2v_embedding_df = pd.read_csv(n2v_emb_file, sep=' ')
    n2v_embedding_df = norm_embeddings(n2v_embedding_df, fname=n2v_emb_file_norm)

    # load normalized embeddings
    n2v_emb = np.loadtxt(n2v_emb_file_norm)
    assert n2v_emb.shape[0] == n2v_embedding_df.shape[0], "Embedding dataframe and normalized version do not match."

    # run k-means on node2vec embeddings
    if "kmeans" in clustering:
        if not plot_db_index:
            n2v_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(n2v_emb)
        else:
            db_search = db_index_search(
                emb=n2v_emb,
                min_clusters=2,
                max_clusters=DB_MAX_CLUSTERS,
                plot_scores=True,
                title_suffix=" - Node2Vec",
                plot_id="n2v",
            )
            n2v_kmeans = db_search["best_kmeans"]
            joblib.dump(db_search, os.path.join(DIR_CACHE, "n2v_db_search"), compress=4)

        # augment dataframe with the clusters found
        n2v_embedding_df.insert(
            loc=len(n2v_embedding_df.columns),
            column='n2v_kmeans',
            value=n2v_kmeans.labels_,
            allow_duplicates=True,
        )

    if "dbscan" in clustering:

        # find optimal number of neighbors in cluster
        method_id = "n2v"
        _, epsilon = find_elbow(n2v_emb, knee_id=method_id)
        print("Elbow: {}".format(epsilon))

        dbscan = DBSCAN(eps=epsilon, metric="euclidean", min_samples=5).fit(n2v_emb)

        # augment dataframe with the clusters found
        n2v_embedding_df.insert(
            loc=len(n2v_embedding_df.columns),
            column='n2v_dbscan',
            value=dbscan.labels_,
            allow_duplicates=True,
        )

    print(n2v_embedding_df.head(10))
    return n2v_embedding_df


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


def dw_cluster(clustering="kmeans", plot_db_index=True):

    clustering = [ clustering ] if isinstance(clustering, str) else clustering

    # load node email-id lookup
    lookup_file = os.path.join(DIR_EMBEDDINGS, "deepwalk_node_lookup.bin")
    emails_to_node_ids = joblib.load(lookup_file)
    node_ids_to_emails = { node_id: email for email, node_id in emails_to_node_ids.items() }

    # load deepwalk embeddings
    # dw_emb_fname = os.path.join(DIR_EMBEDDINGS, "deepwalk.emb")
    dw_emb_fname = os.path.join(DIR_EMBEDDINGS, "dw_comparable.emb")
    dw_df = pd.read_csv(dw_emb_fname, skiprows=1, sep=" ", names=["node_id", "emb_x", "emb_y"])
    dw_df.insert(0, 'node_name', dw_df["node_id"].apply(lambda x: node_ids_to_emails[x]))

    # extract embeddings only
    dw_emb = dw_df[["emb_x", "emb_y"]].to_numpy()

    if "kmeans" in clustering:

        if not plot_db_index:
            dw_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(dw_emb)
        else:
            db_search = db_index_search(
                emb=dw_emb,
                min_clusters=2,
                max_clusters=DB_MAX_CLUSTERS,
                plot_scores=True,
                title_suffix=" - DeepWalk",
                plot_id="dw",
            )
            dw_kmeans = db_search["best_kmeans"]
            joblib.dump(db_search, os.path.join(DIR_CACHE, "dw_db_search"), compress=4)

        # augment dataframe with the clusters found
        dw_df.insert(
            loc=len(dw_df.columns),
            column='dw_kmeans',
            value=dw_kmeans.labels_,
            allow_duplicates=True,
        )

    if "dbscan" in clustering:

        # find optimal number of neighbors in cluster
        method_id = "dw"
        _, epsilon = find_elbow(dw_emb, knee_id=method_id)
        print("Elbow: {}".format(epsilon))

        dbscan = DBSCAN(eps=epsilon, metric="euclidean", min_samples=5).fit(dw_emb)

        # augment dataframe with the clusters found
        dw_df.insert(
            loc=len(dw_df.columns),
            column='dw_dbscan',
            value=dbscan.labels_,
            allow_duplicates=True,
        )

    print(dw_df.head(10))
    return dw_df


def main():

    # enron_path = os.path.join(DIR_ENRON_DB, "emails_cleaned_eric.csv")
    enron_path = os.path.join(DIR_ENRON_DB, "enron_truncated_smallestest.csv")
    enron_graph = load_enron_as_graph(db_path=enron_path)

    clustering = {
        "kmeans": { "name": "K-means" },
        "dbscan": { "name": "DBSCAN" },
    }
    embeddings = {
        "n2v": {
            "name": "Node2Vec",
            "callable": n2v_cluster,
            "args": {
                "data_graph": enron_graph,
                "clustering": list(clustering.keys()),
            },
        },
        "sc": {
            "name": "Spectral Clustering",
            "callable": sc_cluster,
            "args": {
                "data_graph": enron_graph,
                "clustering": list(clustering.keys()),
            },
        },
        "dw": {
            "name": "Deepwalk",
            "callable": dw_cluster,
            "args": {
                "clustering": list(clustering.keys()),
            },
        },
    }

    for method_id, emb_method in embeddings.items():

        # run embedding and clustering methods
        print(" >>> EMBEDDING: {}".format(emb_method["name"]))
        embedding_df = emb_method["callable"](**emb_method["args"])

        csv_out = os.path.join(DIR_PREPROCESSED, "{}_df.csv".format(method_id))
        df_temp = pd.read_csv(csv_out)

        # only save the new dataframe if it has more columns or more rows than the existing one
        more_cols = df_temp.shape[1] < embedding_df.shape[1]
        same_cols = df_temp.shape[1] == embedding_df.shape[1]
        same_or_more_rows = df_temp.shape[0] <= embedding_df.shape[0]
        if more_cols or (same_cols and same_or_more_rows):
            embedding_df.to_csv(csv_out)
        else:
            print("\tnot saving new dataframe, as it appears to have less information than the current one.")

        for cl_id, cl in clustering.items():

            cluster_col = "{}_{}".format(method_id, cl_id)
            print(" >>> \tembedding: {:<15}\tclustering: {}".format(emb_method["name"], cl["name"]))
            # print(embedding_df[cluster_col].value_counts())

            plot_everything(
                graph=enron_graph,
                embedding_df=embedding_df,
                meta={
                    "embedding": emb_method["name"],
                    "clustering": cl["name"],
                },
                # plot_scatter=False,
                # plot_graph=False,
                cluster_col=cluster_col,
            )

        plot_db_scores()


if __name__ == "__main__":
    main()
