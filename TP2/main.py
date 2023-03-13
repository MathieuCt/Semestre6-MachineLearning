import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def part1():
    A = distEuc1(M, Tab)
    print(A)
    data = pd.read_csv("clustering.csv")
    print(data.head())
    N_FEATURE = 2
    data = make_blobs(
        n_samples=100,  # nombre de points
        n_features=2,  # nombre de caractéristiques
        centers=3,  # nombre de clusters
        cluster_std=1.0,  # dispersion des clusters
        center_box=(-5.0, 5.0),
        shuffle=True,
        random_state=11,  # A préciser pour retrouver le même
        # ensemble de point à chaque exécution

        print(data)
    # display clusters
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1])
    plt.show()
    )
def distEuc1(M, Tab):
    return np.sqrt(np.sum((M - Tab)**2, axis = 1))


#4
def initialize_centroids(k, data):
    centroids = data.sample(n = k).values
    centroids = pd.DataFrame(centroids, columns = data.columns)
    return centroids

centroids = initialize_centroids(3, dfB)
print(centroids)

#5
def assign_centroid(data, centroids):
    n_observations = data.shape[0]
    print(n_observations)
    centroid_assign = []
    centroid_distance = []
    centroid_errors = []
    k = centroids.shape[0]
    for observation in range(n_observations):
        dist = DistEuc1(data.iloc[observation,:2], centroids)
        closest_centroid = np.argmin(dist)
        centroid_distance = np.amin(dist)

        #Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_distance)

    return (centroid_assign, centroid_errors)


if __name__ == '__main__':


#c
dfB = pd.DataFrame(data[0], columns = [f"f{i+1}" for i in range(N_FEATURE)])
dfB["y"] = data[1]
print(dfB.head())
print(dfB.shape)

#3
M = np.array((3,4))
Tab = np.array([[1,5], [11,2], [7,5]])


centroid_assign, centroid_errors = assign_centroid(dfB, centroids)
print(centroid_assign)
print(centroid_errors)

