import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np
import sklearn
import node2vec
from sklearn.cluster import KMeans
from node2vec import Node2Vec

email_df = pd.read_csv('./data/emails_cleaned_Eric.csv')
print(email_df)
subset = email_df[0:10000]

G = nx.from_pandas_edgelist(subset, 'From', 'To', edge_attr=['Dates'])
node2vec = Node2Vec(G, dimensions=2, walk_length=20, num_walks=10,workers=4)
# Learn embeddings
model = node2vec.fit(window=10, min_count=1)
#model.wv.most_similar('1')
model.wv.save_word2vec_format("embedding.emb") #save the embedding in file embedding.emb
embedding_df = pd.read_csv('embedding.emb')
print(embedding_df)

# X = np.loadtxt("embedding.emb", skiprows=1) # load the embedding of the nodes of the graph
# #print(X)
# # sort the embedding based on node index in the first column in X
# X=X[X[:,0].argsort()];
# #print(X)
# Z=X[0:X.shape[0],1:X.shape[1]]; # remove the node index from X and save in Z
#
# kmeans = KMeans(n_clusters=2, random_state=0).fit(Z) # apply kmeans on Z
# labels=kmeans.labels_  # get the cluster labels of the nodes.
# print(labels)
# adj_mat = nx.to_numpy_matrix(G)
# print(adj_mat)
# node_list = list(G.nodes())
# print(node_list)
# print(len(node_list))
# lab_map = {lab: i for i, lab in enumerate(sorted(np.unique(node_list)))}
# print(lab_map)
# labels = []
# for idx, label in enumerate(node_list):
#     labels.append(lab_map[label])
# print(labels)
# clusters = SpectralClustering(affinity = 'precomputed', assign_labels="discretize",random_state=0,n_clusters=5).fit_predict(adj_mat)
# plt.scatter(labels,clusters,c=clusters, s=50, cmap='viridis')
# plt.show()
#
# plt.figure(figsize=(20,20))
# pos = nx.spring_layout(G, k=.1)
# nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=False, edge_color='blue')
# plt.show()

