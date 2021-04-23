import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans, SpectralClustering

import node2vec
from node2vec import Node2Vec


DIR_DATASETS_BASE = "/media/Data/datasets/graphs/"
DIR_ENRON_DB = os.path.join(DIR_DATASETS_BASE, "enron/preprocessed")
RANDOM_STATE = 42


def load_graph(db_path):

    email_df = pd.read_csv(db_path)
    print(email_df)
    subset = email_df[0:10000]

    graph = nx.from_pandas_edgelist(subset, 'From', 'To', edge_attr=['Dates'])

    return graph


def run_node2vec(graph, emb_file='embedding.emb'):

    node2vec = Node2Vec(graph, dimensions=2, walk_length=20, num_walks=10, workers=4)

    # learn embeddings
    model = node2vec.fit(window=10, min_count=1)
    # model.wv.most_similar('1')

    # save outputs
    model.wv.save_word2vec_format(emb_file)

    return model


def norm_embeddings(embedding_df, write_to_csv=True):
    """Converts email addresses in embedding to numeric IDs."""

    embedding_df = embedding_df.reset_index()
    embedding_df.insert(0, 'node_id', range(0, len(embedding_df)))
    embedding_df.rename(
        inplace=True,
        columns={
            embedding_df.columns[-2]: "emb_x",      # second to last
            embedding_df.columns[-1]: "emb_y"       # last
        },
    )
    selected_cols = ["node_id", "emb_x", "emb_y"]

    if write_to_csv:
        emb_file_normalized = 'embedding_norm.emb'
        embedding_df[selected_cols].to_csv(emb_file_normalized, header=False, sep=' ')

    return embedding_df, emb_file_normalized


def run_spectral_clustering(graph):

    adj_mat = nx.to_numpy_matrix(graph)
    node_list = list(graph.nodes())
    unique_list = list(set(node_list))
    print(len(node_list), len(unique_list))
    emails_to_node_ids = {
        email: node_id \
        for node_id, email in enumerate(sorted(unique_list))
    }
    node_ids = emails_to_node_ids.values()

    clusters = SpectralClustering(
        affinity='precomputed',
        assign_labels="discretize",
        random_state=RANDOM_STATE,
        n_clusters=5
    ).fit_predict(adj_mat)

    return {
        "clusters": clusters,
        "node_ids": node_ids,
    }


def plot_scatter(graph=None, node_ids=None, clusters=None):
    """Create plots given a clusters and node_ids, and/or a graph."""

    if all(x is not None for x in [clusters, node_ids]):
        plt.scatter(node_ids, clusters, c=clusters, s=50, cmap='viridis')
        plt.show()

    if all(x is not None for x in [graph]):
        plt.figure(figsize=(20,20))
        pos = nx.spring_layout(graph, k=.1)
        nx.draw_networkx(graph, pos, node_size=25, node_color='red', with_labels=False, edge_color='blue')
        plt.show()


def main():

    # init
    emb_file = 'embedding.emb'
    enron_path = os.path.join(DIR_ENRON_DB, "emails_cleaned_eric.csv")

    # load graph and get embeddings
    graph = load_graph(db_path=enron_path)
    # n2v = run_node2vec(graph=graph, emb_file=emb_file)

    # reload embeddings and convert their email addresses to numeric IDs
    embedding_df = pd.read_csv(emb_file, sep=' ')
    embedding_df, emb_file_norm = norm_embeddings(embedding_df)

    n2v_emb = np.loadtxt(emb_file_norm, skiprows=1) # load the embedding of the nodes of the graph

    # sort the embedding based on node index in the first column in X
    n2v_emb = n2v_emb[n2v_emb[:,0].argsort()];
    # Z=X[0:X.shape[0],1:X.shape[1]]; # remove the node index from X and save in Z

    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(n2v_emb) # apply kmeans on Z
    labels = kmeans.labels_
    print(labels)

    sc_results = run_spectral_clustering(graph)
    plot_scatter(
        # graph=graph,
        node_ids=sc_results["node_ids"],
        clusters=sc_results["clusters"],
    )


if __name__ == "__main__":
    main()
