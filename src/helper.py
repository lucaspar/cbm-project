import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import networkx as nx

from kneed import KneeLocator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score


warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

class Experiment():

    def __init__(self, exp_type, dir_dataset_base="../data/", random_state=3001):
        """
            Class that encapsulates methods and parameters referent to a group of similar procedures
            that are connected by a set of parameters (dataset, partition size, random seed, etc) that
            form an experiment. These procedures can be for example:
                + analyses and metrics (e.g. parameter search, model evaluation);
                + creation of plots (e.g. scatter, bar, timelines);
                + creation of other artifacts (e.g. embeddings, preprocessed dataframes).
        """

        # params for inputs
        ds_split = exp_type.split("_")
        self.DB_NAME, self.DB_VARIATION = ds_split[0], "_".join(ds_split[1:])
        self.DIR_DATASETS_BASE = os.path.normpath(dir_dataset_base)
        self.DIR_DB = os.path.join(self.DIR_DATASETS_BASE, self.DB_NAME)
        self.SUBSET_SIZE = -10_000
        FULL_OR_SUBSET_SIZE = self.SUBSET_SIZE if self.SUBSET_SIZE > 0 else "full"
        self.db_path = None

        # experiment-dependent params
        if exp_type == "enron_comparable":
            self.db_path = os.path.join(self.DIR_DB, "enron_truncated_smallestest.csv")
            self.DW_EMBEDDING = "dw_comparable.emb"
            self.N2V_EMBEDDING = "n2v_embedding_{}.emb".format(exp_type)

        elif exp_type == "enron_full":
            self.db_path = os.path.join(self.DIR_DB, "emails_cleaned_eric.csv")
            self.DW_EMBEDDING = "deepwalk.emb"
            # self.N2V_EMBEDDING = "n2v_embedding_full_ericcsv.emb"
            self.N2V_EMBEDDING = 'n2v_embedding_{}_{}.emb'.format(exp_type, str(FULL_OR_SUBSET_SIZE))

        elif exp_type in ["synth_p001_a003", "synth_p00025_a005"]:
            self.DW_EMBEDDING = "dw_{}.emb".format(exp_type)
            self.N2V_EMBEDDING = 'n2v_{}_{}.emb'.format(exp_type, str(FULL_OR_SUBSET_SIZE))

        else:
            raise ValueError("Unknown experiment type: {}".format(exp_type))

        print(" >>> EXP TYPE: {}".format(exp_type))

        # params for outputs
        self.DIR_OUTPUTS_BASE = "outputs"
        self.DIR_PLOTS = os.path.join(self.DIR_OUTPUTS_BASE, "plots_{}".format(exp_type))
        self.DIR_EMBEDDINGS = os.path.join(self.DIR_OUTPUTS_BASE, "embeddings".format(exp_type))
        self.DIR_PREPROCESSED = os.path.join(self.DIR_EMBEDDINGS, "preprocessed_{}".format(exp_type))
        self.DIR_CACHE = "./cache"
        self.OUT_PLOTS_EXTENSION = "png"    # svg | pdf | png
        self.RANDOM_STATE = random_state
        self.COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

        # create output dirs
        new_dir_list = [self.DIR_CACHE, self.DIR_PLOTS, self.DIR_PREPROCESSED]
        self.create_dirs(new_dir_list)


    def create_dirs(_, new_dir_list):
        for d in new_dir_list:
            os.makedirs(d, exist_ok=True)


    def load_synthetic_dataset(self, exp_type):
        """
            Loads the synthetically generated dataset returning two dataframes:
            one with the edge list, the other with the node ground-truth.
        """

        db_edges = os.path.join(self.DIR_DB, "{}.edgelist".format(self.DB_VARIATION))
        db_nodes = os.path.join(self.DIR_DB, "{}.nodes".format(self.DB_VARIATION))
        df_edges = pd.read_csv(db_edges, sep="\s+", header=None, names=["From", "To", "Timestamp"])
        df_nodes = pd.read_csv(db_nodes, sep="\s+", header=None, names=["Node", "Anomaly"])

        return df_edges, df_nodes


    def load_edgelist_as_graph(self, db_path=None, df=None, edge_attr=["Dates"], names=None):
        """Loads pre-processed Enron dataset as a graph."""

        assert any([x is not None for x in [db_path, df]]),\
            "'db_path' or 'df' must be passed to this method."

        if df is None:

            # load dataset as a dataframe
            print("Loading {}...".format(db_path))
            df = pd.read_csv(db_path, names=names)

            # remove spaces from emails
            for col in ["From", "To"]:
                df[col] = df[col].apply(lambda x: x.replace(" ", ""))

        else:
            print("Using dataset loaded in dataframe...")

        # build graph
        subset = df[0:self.SUBSET_SIZE] if self.SUBSET_SIZE > 0 else df
        graph = nx.from_pandas_edgelist(subset, 'From', 'To', edge_attr=edge_attr)

        return graph


    def store_embedding_df(_, df, dir_out=".", df_id="emb", overwrite_mode='smart', verbose=True):
        """
            Stores an embedding file on disk only. If the file already exists,
            it analyzes the existing one to determine whether it should overwrite its contents or not.
            Args:
                overwrite_mode (str, False): "smart" overwrites dataframe when new dataframe has the same shape or it is larger,
                        "force" will always overwrite an existing dataframe (info in memory persists),
                        False disables any overwrite (info already on disk is kept)
            Returns:
                True if dataframe was stored on disk.
        """

        assert overwrite_mode in ["smart", "force", False], "Overwrite mode is invalid: '{}'".format(overwrite_mode)

        csv_out = os.path.join(dir_out, "{}_df.csv".format(df_id))
        if os.path.exists(csv_out):

            if not overwrite_mode:
                if verbose:
                    print("Output embeddings file exists and overwriting is disabled. The contents were not stored on disk.")
                return False

            df_temp = pd.read_csv(csv_out)

            # only save the new dataframe if it has more columns or more rows than the existing one
            more_cols = df_temp.shape[1] < df.shape[1]
            same_cols = df_temp.shape[1] == df.shape[1]
            same_or_more_rows = df_temp.shape[0] <= df.shape[0]

            # the new dataframe seems to have new contents
            if (overwrite_mode == "force") or \
                (overwrite_mode == "smart" and (more_cols or (same_cols and same_or_more_rows))):
                df.to_csv(csv_out)
                if verbose:
                    print("Dataframe saved as '{}'".format(csv_out))
                return True

            # new dataframe seems to have less information, do not overwrite it
            if verbose:
                print("\t >>> WARNING :: Not saving new dataframe, as it appears to have " + \
                    "less information than the current one in:\n\t{}".format(csv_out))
            return False

        else:
            if verbose:
                print("Embedding dataframe not found, creating it...")
            df.to_csv(csv_out)
            return True


    def find_elbow(
            self,
            embedding,
            knee_id="",
            n_neighbors=4,
            plot=True,
        ):
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
        os.makedirs(os.path.join(self.DIR_PLOTS, PLT_SUBPLOT), exist_ok=True)
        knee_fname = os.path.join(self.DIR_PLOTS, PLT_SUBPLOT, "{}.{}".format(knee_id, self.OUT_PLOTS_EXTENSION))
        plt.savefig(knee_fname)
        plt.close()
        print("Saved knee plot as '{}'".format(knee_fname))

        return knee_x, knee_y


    def db_index_search(self, emb, min_clusters=2, max_clusters=32, plot_scores=False, title_suffix="", plot_id=None):
        """
            Performs a search for the optimal number of clusters in an embedded space using Davies-Bouldin index.
            Returns:
                best_k:         optimal number of clusters by computing Davies-Bouldin index.
                best_db_index:  optimal Davies-Bouldin index achieved by `best_k`.
                best_kmeans:    the optimal clustering.
                scores:         all DB indices computed as a dictionary { "k": "DB index" }.
        """

        PLT_SUBDIR = "db-index"
        os.makedirs(os.path.join(self.DIR_PLOTS, PLT_SUBDIR), exist_ok=True)

        best_k = min_clusters
        best_db_index = np.inf
        scores = dict()
        for current_k in range(min_clusters, max_clusters + 1, 1):
            kmeans = KMeans(n_clusters=current_k, random_state=self.RANDOM_STATE).fit(emb)
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
            fname = os.path.join(self.DIR_PLOTS, PLT_SUBDIR, "{}.{}".format(plot_id, self.OUT_PLOTS_EXTENSION))
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


    def plot_db_scores(self):
        """Plots DB score curves in the same plot."""

        PLT_SUBDIR = "db-index"
        os.makedirs(os.path.join(self.DIR_PLOTS, PLT_SUBDIR), exist_ok=True)
        fname = os.path.join(self.DIR_PLOTS, PLT_SUBDIR, "all-scores.{}".format(self.OUT_PLOTS_EXTENSION))

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
            fpath = os.path.join(self.DIR_CACHE, fname)
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


    def add_value_labels(_, ax, spacing=5, fontsize=8):
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


    def plot_everything(self, graph=None, embedding_df=None, clusters=None, meta="", plot_graph=True, plot_scatter=True, cluster_col="cluster"):
        """Create plots given a clusters and node_ids, and/or a graph."""

        assert [x in meta for x in ["embedding", "clustering"]]
        embedding_df = embedding_df.copy()
        embedding_df = embedding_df.set_index("node_name")

        # assign colors to the clusters
        color_map, cluster_freqs = dict(), dict()
        frequencies_iter = embedding_df[cluster_col].value_counts().iteritems()
        for idx, (cluster, freq) in enumerate(frequencies_iter):
            color = self.COLORS[idx % len(self.COLORS)]
            color_map[cluster] = color
            cluster_freqs[cluster] = freq
            # print("\tcluster={:<4}\tfreq={:<4}\tcolor={}".format(cluster, freq, color))
        if len(self.COLORS) <= len(color_map):
            print(" >>> WARNING: more clusters than colors available.")

        # scatter plot of nodes by cluster
        if plot_scatter and all(x is not None for x in [embedding_df]):

            PLT_SUBDIR = "scatter"
            os.makedirs(os.path.join(self.DIR_PLOTS, PLT_SUBDIR), exist_ok=True)
            fname = os.path.join(self.DIR_PLOTS, PLT_SUBDIR, "{}.{}".format(cluster_col, self.OUT_PLOTS_EXTENSION))

            # subplots
            fig, axs = plt.subplots(2, figsize=(6, 6), gridspec_kw={'height_ratios': [4, 1]})
            axs = axs.flatten()
            ax = axs[0]

            marker_size = 5 / np.log10(embedding_df["emb_x"].shape[0]) + 4
            ax = embedding_df.plot.scatter("emb_x", "emb_y",
                ax=ax,
                color=embedding_df[cluster_col].map(color_map),
                s=marker_size,
                # cmap='viridis',
            )
            ax.set_title("{} {}".format(meta["embedding"], meta["clustering"]))

            # create legend for outliers
            if -1 in color_map:
                outlier_color = color_map[-1] if -1 in color_map else None
                outlier_patch = mpatches.Patch(color=outlier_color, label='DBSCAN Outliers')
                ax.legend(handles=[outlier_patch])
                print("]\t\t>> OUTLIER color: {}".format(outlier_color))

            ax = axs[1]
            x = cluster_freqs.keys()
            heights = cluster_freqs.values()
            print("\t\t >> Outlier frequency: {}".format(cluster_freqs[-1] if -1 in cluster_freqs else 0))
            ax.bar(
                x, height=heights,
                color=list(color_map.values()),
            )
            self.add_value_labels(ax, fontsize=6)
            ax.title.set_text('Frequencies by cluster')

            fig.tight_layout()
            fig.savefig(fname)
            print("Saved scatter plot as '{}'".format(fname))

        # graph plot color-coded by cluster
        if plot_graph and all(x is not None for x in [graph, embedding_df]):

            PLT_SUBDIR = "graph"
            os.makedirs(os.path.join(self.DIR_PLOTS, PLT_SUBDIR), exist_ok=True)

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

                fname = os.path.join(self.DIR_PLOTS, PLT_SUBDIR, "{}_{}.{}".format(cluster_col, layout_style, self.OUT_PLOTS_EXTENSION))

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
