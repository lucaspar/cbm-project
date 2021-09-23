#!/bin/env python
import os

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import davies_bouldin_score, classification_report

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

    # print(sc_df.head(10))
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

    # print(n2v_embedding_df.head(10))
    return n2v_embedding_df


def dw_cluster(emb_fname, node_map_fname=None, clustering="kmeans", exp=None, plot_db_index=True, cluster_gt=None):
    """
        Loads precomputed deepwalk embeddings and runs the clustering on them.
        emb_fname (str):        path to file containing the list of node IDs coordinates in the X,Y embedding space.
        node_map_fname (str):   path to file containing a dictionary with { node_id: node_name } key-value pairs.
    """

    clustering = [ clustering ] if isinstance(clustering, str) else clustering

    # load deepwalk embeddings
    print(" >>> Reading Deepwalk embedding: '{}'".format(emb_fname))
    dw_df = pd.read_csv(emb_fname, skiprows=1, sep=" ", names=["node_id", "emb_x", "emb_y"])

    # load node email-id lookup
    if node_map_fname is not None:
        node_names_to_node_ids = joblib.load(node_map_fname)
        node_ids_to_node_names = { node_id: name for name, node_id in node_names_to_node_ids.items() }
    # or use node IDs as their names
    else:
        print("Node map not specified, using node IDs as their names.")
        node_ids_to_node_names = { node_id: node_id for node_id in dw_df["node_id"] }

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

    # print(dw_df.head(10))
    return dw_df


def anomaly_supervised_eval(df_pred, df_gt, pred_col="clustering", ANOMALY_LABEL=-1, gt_col="anomaly", pred_id="node_id", gt_id="Node", title=None):
    """Given a prediction and a ground-truth dataframe, runs a supervised evaluation of the anomaly predictions."""

    print(df_pred.head())
    print(df_gt.head())
    print(pred_id, gt_id)
    merged = pd.merge(left=df_pred, right=df_gt, left_on=pred_id, right_on=gt_id)
    # print(merged.head(500))

    gt_series = merged[gt_col].apply(lambda x: "Anomaly" if x == 1 else "Normal")
    pred_series = merged[pred_col].apply(lambda x: "Anomaly" if x == ANOMALY_LABEL else "Normal")
    # pred_series = merged[pred_col].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    print("Ground-truth frequencies:")
    print(gt_series.value_counts())
    # print(df_gt[gt_col].value_counts())
    print("Prediction frequencies:")
    print(pred_series.value_counts())
    # print(df_pred[pred_col].value_counts())
    print()

    if title:
        print("\n\t ::: {} :::".format(title))
    print(classification_report(
        y_true=gt_series,
        y_pred=pred_series,
    ))


def run_experiment(exp_type, db_dir="../data/"):

    exp = helper.Experiment(
        exp_type,
        dir_dataset_base=db_dir,
        random_state=RANDOM_STATE,
    )

    # vars init
    df_nodes_gt = None
    dw_node_map = None

    # real dataset
    if exp.db_path is not None:
        ds_graph = exp.load_edgelist_as_graph(db_path=exp.db_path, names=["Dates", "From", "To"])
        if exp_type.startswith("enron_"):
            dw_node_map = os.path.join(exp.DIR_EMBEDDINGS, "deepwalk_node_lookup.bin")
    # synthetic dataset
    else:
        df_edges, df_nodes_gt = exp.load_synthetic_dataset()
        ds_graph = exp.load_edgelist_as_graph(df=df_edges, edge_attr=["Timestamp"])

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
                "exp": exp,
            },
            "cluster_gt": df_nodes_gt,
        },
        "sc": {
            "name": "Spectral Clustering",
            "callable": sc_cluster,
            "args": {
                "data_graph": ds_graph,
                "clustering": list(clustering.keys()),
                "exp": exp,
            },
            "cluster_gt": df_nodes_gt,
        },
        "dw": {
            "name": "Deepwalk",
            "callable": dw_cluster,
            "args": {
                "emb_fname": os.path.join(exp.DIR_EMBEDDINGS, exp.DW_EMBEDDING),
                "node_map_fname": dw_node_map,
                "clustering": list(clustering.keys()),
                "exp": exp,
            },
            "cluster_gt": df_nodes_gt,
        },
    }

    for emb_method_id, payload in embeddings.items():

        # run embedding and clustering methods
        print(" >>> EMBEDDING: {}".format(payload["name"]))
        embedding_df = payload["callable"](**payload["args"])

        exp.store_embedding_df(
            df=embedding_df,
            dir_out=exp.DIR_PREPROCESSED,
            df_id=emb_method_id,
        )

        for cl_id, cl in clustering.items():

            cluster_col = "{}_{}".format(emb_method_id, cl_id)
            print(" >>> \tembedding: {:<15}\tclustering: {}".format(payload["name"], cl["name"]))
            # print(embedding_df[cluster_col].value_counts())

            if payload["cluster_gt"] is not None:
                if cl_id == "dbscan":
                    print("Running supervised evaluation...")
                    anomaly_supervised_eval(
                        df_pred=embedding_df,
                        df_gt=payload["cluster_gt"],
                        pred_col=cluster_col,
                        gt_col="Anomaly",
                        title=payload["name"],
                    )
                else:
                    print("Skipping supervised evaluation for clustering != 'dbscan'")
            else:
                print("Anomaly ground-truth not found, skipping supervised evaluation")

            exp.plot_everything(
                graph=ds_graph,
                embedding_df=embedding_df,
                meta={
                    "embedding": payload["name"],
                    "clustering": cl["name"],
                },
                # plot_scatter=False,
                plot_graph=False,
                cluster_col=cluster_col,
            )

        exp.plot_db_scores()


def plot_scatter(embedding_df, clustering_df, colors):

    color_map, cluster_freqs = dict(), dict()
    frequencies_iter = clustering_df['cluster'].value_counts().iteritems()
    for idx, (cluster, freq) in enumerate(frequencies_iter):
        color = colors[idx % len(colors)]
        color_map[cluster] = color
        cluster_freqs[cluster] = freq
    x = cluster_freqs.keys()
    heights = cluster_freqs.values()

    fig, ax = plt.subplots()
    marker_size = 5 / np.log10(embedding_df["emb_x"].shape[0]) + 4
    ax = embedding_df.plot.scatter("emb_x", "emb_y",
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
    # add_value_labels(ax, fontsize=6)
    fig.savefig('bars.pdf')

    return


def eval_egonet(exp_type, db_dir):

    exp = helper.Experiment(
        exp_type,
        dir_dataset_base=db_dir,
        random_state=RANDOM_STATE,
    )

    embeddings = {}
    with open('../embeddings/synthetic_clique_{}.emb'.format(exp.DB_VARIATION)) as fp:
        for line in fp:
            node, vector = line.strip().split('\t')
            embeddings[int(node)] = list(map(float, vector.replace(',', '').replace('[', '').replace(']', '').split(' ')))

    egonet_emb = np.asarray([vector for vector in embeddings.values()])
    _, y = exp.find_elbow(egonet_emb)

    from sklearn.preprocessing import RobustScaler

    # reduce dimensions
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    egonet_emb = scaler.fit_transform(egonet_emb)
    pca = PCA(n_components=2)
    Y = pca.fit_transform(egonet_emb)
    embedding_df = pd.DataFrame(egonet_emb)

    # apply DBSCAN to label outliers
    clustering = DBSCAN(eps=y, min_samples=2).fit(egonet_emb)
    clustering_dict = { 'node': list(range(len(clustering.labels_))), 'cluster': clustering.labels_ }
    clustering_df = pd.DataFrame(clustering_dict)

    print(clustering_df.cluster.value_counts())

    pca_df = pd.DataFrame(Y)
    pca_df.columns = ['emb_x', 'emb_y']
    # pca_df.columns = ['emb_x', 'emb_y', 'emb_z']
    pca_df.reset_index(level=0, inplace=True)

    print(clustering_df.tail())
    print(pca_df.tail())
    final_df = pd.merge(
        left=clustering_df, left_on='node',
        right=pca_df,       right_on='index',
        how="outer",
    )
    print(final_df.head())

    _, df_nodes_gt = exp.load_synthetic_dataset()

    anomaly_supervised_eval(
        df_pred=clustering_df,
        df_gt=df_nodes_gt,
        ANOMALY_LABEL=1,
        pred_col="cluster",
        pred_id="node",
        gt_id="Node",
        gt_col="Anomaly",
        title="Egonet",
    )

    plot_scatter(pca_df, clustering_df, colors=exp.COLORS)


def main():

    experiments = [
        # "enron_full",
        # "enron_comparable",
        "synth_p001_a003",
        # "synth_p00025_a005",
    ]

    # run baseline experiments
    for exp_type in experiments:

        print("")
        run_experiment(exp_type, db_dir="../data/")
        print("\n------------\n\n")

        # run evaluation on egonet
        eval_egonet(exp_type, db_dir="../data/")


if __name__ == "__main__":
    main()
