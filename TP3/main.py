import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


def ex1():
    # create 3 clusters with 100 points
    N_FEATURE = 2
    X, Y = make_blobs(
        n_samples=30,  # nombre de points
        n_features=2,  # nombre de caractéristiques
        centers=3,  # nombre de clusters
        cluster_std=1.0,  # dispersion des clusters
        center_box=(-5.0, 5.0),
        shuffle=False,
        random_state=12
        ,  # A préciser pour retrouver le même
        # ensemble de point à chaque exécution
    )
    # display data set
    # set subplot
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(X[:, 0], X[:, 1], c=Y)
    # Use the AgglomerativeClustering for CHA clustering
    agg = AgglomerativeClustering(n_clusters=3, linkage='average', distance_threshold=None)
    # Display the clustering result
    # axs[1].scatter(X[:, 0], X[:, 1], c=Y)

    agg = agg.fit_predict(X)
    # Display the clustering result
    axs[1].scatter(X[:, 0], X[:, 1], c=agg)
    plt.show()


def ex2():
    X, Y = make_blobs(
        n_samples=30,  # nombre de points
        n_features=2,  # nombre de caractéristiques
        centers=3,  # nombre de clusters
        cluster_std=1.0,  # dispersion des clusters
        center_box=(-5.0, 5.0),
        shuffle=False,
        random_state=12
        ,  # A préciser pour retrouver le même
        # ensemble de point à chaque exécution
    )
    Z = linkage(X, 'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, labels=Y, leaf_rotation=90, leaf_font_size=6)
    Z = linkage(X, 'single')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, labels=Y, leaf_rotation=90, leaf_font_size=6)

    plt.show()


if __name__ == '__main__':
    # ex1()
    ex2()
