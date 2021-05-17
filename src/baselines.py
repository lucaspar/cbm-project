#!/bin/env python
import os

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

import helper


# global params
RANDOM_STATE = 42           # seed to be used in pseudorandom operations


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


def sc_cluster(data_graph, clustering=["kmeans"], plot_db_index=True, exp=None, cluster_gt=None):
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

        if plot_db_index and exp is not None:
            db_search = exp.db_index_search(
                emb=sc_emb,
                min_clusters=2,
                plot_scores=True,
                title_suffix=" - Spectral Clustering",
                plot_id="sc",
            )
            sc_kmeans = db_search["best_kmeans"]
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
        _, epsilon = exp.find_elbow(sc_emb, knee_id=method_id)
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


def n2v_cluster(emb_fname, data_graph, clustering=["kmeans"], exp=None, plot_db_index=True, cluster_gt=None):
    """Runs node2vec embedding + clustering on dataset."""

    clustering = [ clustering ] if isinstance(clustering, str) else clustering

    # init
    n2v_emb_file_norm = "n2v_norm.emb"

    # load graph and get embeddings
    if not os.path.exists(emb_fname):
        _ = run_node2vec(graph=data_graph, emb_out_file=emb_fname)

    # reload embeddings and convert their email addresses to numeric IDs
    n2v_embedding_df = pd.read_csv(emb_fname, sep=' ')
    n2v_embedding_df = norm_embeddings(n2v_embedding_df, fname=n2v_emb_file_norm)

    # load normalized embeddings
    n2v_emb = np.loadtxt(n2v_emb_file_norm)
    assert n2v_emb.shape[0] == n2v_embedding_df.shape[0], "Embedding dataframe and normalized version do not match."

    # run k-means on node2vec embeddings
    if "kmeans" in clustering:
        if not plot_db_index:
            n2v_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(n2v_emb)
        elif exp is not None:
            db_search = exp.db_index_search(
                emb=n2v_emb,
                min_clusters=2,
                plot_scores=True,
                title_suffix=" - Node2Vec",
                plot_id="n2v",
            )
            n2v_kmeans = db_search["best_kmeans"]

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
        _, epsilon = exp.find_elbow(n2v_emb, knee_id=method_id)
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


def dw_cluster(emb_fname, node_map_fname, clustering="kmeans", exp=None, plot_db_index=True, cluster_gt=None):
    """
        Loads precomputed deepwalk embeddings and runs the clustering on them.
        emb_fname (str):        path to file containing the list of node IDs coordinates in the X,Y embedding space.
        node_map_fname (str):   path to file containing a dictionary with { node_id: node_name } key-value pairs.
    """

    clustering = [ clustering ] if isinstance(clustering, str) else clustering

    # load node email-id lookup
    node_names_to_node_ids = joblib.load(node_map_fname)
    node_ids_to_node_names = { node_id: name for name, node_id in node_names_to_node_ids.items() }

    # load deepwalk embeddings
    dw_df = pd.read_csv(emb_fname, skiprows=1, sep=" ", names=["node_id", "emb_x", "emb_y"])
    dw_df.insert(0, 'node_name', dw_df["node_id"].apply(lambda x: node_ids_to_node_names[x]))

    # extract embeddings only
    dw_emb = dw_df[["emb_x", "emb_y"]].to_numpy()

    if "kmeans" in clustering:

        if not plot_db_index:
            dw_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE).fit(dw_emb)
        elif exp is not None:
            db_search = exp.db_index_search(
                emb=dw_emb,
                min_clusters=2,
                plot_scores=True,
                title_suffix=" - DeepWalk",
                plot_id="dw",
            )
            dw_kmeans = db_search["best_kmeans"]

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
        _, epsilon = exp.find_elbow(dw_emb, knee_id=method_id)
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

    # params
    # exp_type = "enron_full"
    # exp_type = "enron_comparable"
    # exp_type = "synth_p001_a003"
    exp_type = "synth_p00025_a005"

    dir_dataset_base = "/media/Data/datasets/graphs/"
    dir_dataset_base = "../data/"
    exp = helper.Experiment(
        exp_type,
        dir_dataset_base=dir_dataset_base,
        random_state=RANDOM_STATE,
    )

    df_edges, df_nodes = exp.load_synthetic_dataset(exp_type=exp_type)
    ds_graph = exp.load_edgelist_as_graph(db_path=exp.db_path) \
        if exp.db_path is not None else exp.load_edgelist_as_graph(df=df_edges, edge_attr=["Timestamp"])

    clustering = {
        "kmeans": { "name": "K-means" },
        "dbscan": { "name": "DBSCAN" },
    }
    embeddings = {
        "n2v": {
            "name": "Node2Vec",
            "callable": n2v_cluster,
            "args": {
                "emb_fname": os.path.join(exp.DIR_EMBEDDINGS, exp.N2V_EMBEDDING),
                "data_graph": ds_graph,
                "clustering": list(clustering.keys()),
                "cluster_gt": df_nodes,
                "exp": exp,
            },
        },
        "sc": {
            "name": "Spectral Clustering",
            "callable": sc_cluster,
            "args": {
                "data_graph": ds_graph,
                "clustering": list(clustering.keys()),
                "cluster_gt": df_nodes,
                "exp": exp,
            },
        },
        "dw": {
            "name": "Deepwalk",
            "callable": dw_cluster,
            "args": {
                "emb_fname": os.path.join(exp.DIR_EMBEDDINGS, exp.DW_EMBEDDING),
                "node_map_fname": os.path.join(exp.DIR_EMBEDDINGS, "deepwalk_node_lookup.bin"),
                "clustering": list(clustering.keys()),
                "cluster_gt": df_nodes,
                "exp": exp,
            },
        },
    }

    for method_id, emb_method in embeddings.items():

        # run embedding and clustering methods
        print(" >>> EMBEDDING: {}".format(emb_method["name"]))
        embedding_df = emb_method["callable"](**emb_method["args"])

        exp.store_embedding_df(
            df=embedding_df,
            dir_out=exp.DIR_PREPROCESSED,
            df_id=method_id,
        )

        for cl_id, cl in clustering.items():

            cluster_col = "{}_{}".format(method_id, cl_id)
            print(" >>> \tembedding: {:<15}\tclustering: {}".format(emb_method["name"], cl["name"]))
            # print(embedding_df[cluster_col].value_counts())

            exp.plot_everything(
                graph=ds_graph,
                embedding_df=embedding_df,
                meta={
                    "embedding": emb_method["name"],
                    "clustering": cl["name"],
                },
                # plot_scatter=False,
                plot_graph=False,
                cluster_col=cluster_col,
            )

        exp.plot_db_scores()


if __name__ == "__main__":
    main()
