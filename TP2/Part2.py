# Machine Learning TP2

# Rappeler le principe de l’algorithme Kmeans :
'''
1. Choisir le nombre de clusters k
2. Choisir k points comme centres de clusters
3. Assigner chaque point au cluster le plus proche
4. Recalculer les centres des clusters
'''

# The goal of this part is to implement the Kmeans from scratch.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.datasets import make_blobs


# read clustering.csv
data = pd.read_csv("clustering.csv")
# create dataframe with "Applic
# antIncome" and "LoanAmount" columns
dfA = data[['ApplicantIncome', 'LoanAmount']]
#print(dfA.head())
# create 3 clusters with 100 points
N_FEATURE = 2
data = make_blobs(
    n_samples=100,  # nombre de points
    n_features=2,  # nombre de caractéristiques
    centers=3,  # nombre de clusters
    cluster_std=1.0,  # dispersion des clusters
    center_box=(-5.0, 5.0),
    shuffle=False,
    random_state=11
    ,  # A préciser pour retrouver le même
    # ensemble de point à chaque exécution
)
# Display generated data using scatter plot
#plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1])
#plt.show()
# Create a dataframe with the generated data and the cluster
data2 = data[0]
data2 = np.append(data2, data[1].reshape(100, 1), axis=1)
dfB = pd.DataFrame(data[0], columns=['x', 'y'])
# Display data frame
# print(dfB)

def DistEucl(M,Tab):
    '''
    Compute the euclidean distance between a point and the centroids
    '''
    dist = []
    # For each centroid
    for i in range(len(Tab)):

        sub = np.subtract(M,Tab[i])
        square = np.square(sub)
        sum = np.sum(square)
        dist.append(np.sqrt(sum))
    return dist

def Initialize_centroids(k,data):
    '''
    Initialize the centroids
    '''
    # Randomly choose k centroids from the data
    centroids = data.sample(n=k,random_state=11 ).values
    centroids = pd.DataFrame(centroids, columns = data.columns)
    return centroids

def assign_centroid(data, centroids):
    '''
    Assign the closest centroid to each point
    '''
    n = data.shape[0]  
    centroid_assign = []
    centroid_distance = []  
    k = centroids.shape[0]
    for i in range(n):
        dist = DistEucl(data.iloc[i,:2].values, centroids.values)
        centroid_assign.append(np.argmin(dist))
        centroid_distance.append(np.min(dist))

    return (centroid_assign,centroid_distance)

def update_centroids(data,centroid_assign, k):
    '''
    Update the centroids
    '''
    centroids = []
    for i in range(k):
        cluster = []
        # Select the points that belong to the cluster i
        for j in range(len(centroid_assign)):
            if centroid_assign[j] == i:
                cluster.append(data.iloc[j,:2])
        # Compute the mean of the points
        # print ("cluster :", np.array(cluster))
        centroids.append(np.array(cluster).mean(axis=0))
    return pd.DataFrame(centroids, columns=data.columns)

fig, axs = plt.subplots(2, 2)
plt.subplots(2, 2)
# Kmeans first iteration
centroidsA = Initialize_centroids(3, dfA)
centroidsB = Initialize_centroids(3, dfB)

# Assign the closest centroid to each point
centroid_assignA, centroid_distanceA = assign_centroid(dfA, centroidsA)
centroid_assignB, centroid_distanceB = assign_centroid(dfB, centroidsB)

# Display the results in a plot
axs[0, 0].scatter(dfA['ApplicantIncome'], dfA['LoanAmount'], c=centroid_assignA)
axs[0, 1].scatter(dfB['x'], dfB['y'], c=centroid_assignB)

# Update the centroids
centroidsA = update_centroids(dfA,centroid_assignA, 3)
centroidsB = update_centroids(dfB,centroid_assignB, 3)
centroid_assignA, centroid_distanceA = assign_centroid(dfA, centroidsA)
centroid_assignB, centroid_distanceB = assign_centroid(dfB, centroidsB)

axs[1,0].scatter(dfA['ApplicantIncome'], dfA['LoanAmount'], c=centroid_assignA)
axs[1,1].scatter(dfB['x'], dfB['y'], c=centroid_assignB)
plt.show()


def ComputeInertieW(data, centroids, cluster):
    """
    Compute the intra-cluster inertia
    """
    # Inertie Totale = Inertie Inter-classe + Inertie Intra-classe
    inertieTotale = 0
    for i in range(len(centroids)):
        # Inertie Intra-classe
        inertieInter = 0
        for j in range(len(cluster)):
            if cluster[j] == i:
                inertieInter += np.sum(np.square(np.subtract(data.iloc[j, :2].values, centroids.iloc[i, :2].values)))
        inertieTotale += inertieInter
    # Calcule du centre de gravité
    G = data.mean(axis=0)
    # Inertie Inter-classe
    inertieInter = 0
    for i in range(len(centroids)):
        inertieInter += len(cluster[cluster == i].shape) * np.sum(np.square(np.subtract(centroids.iloc[i, :2].values, G.values)))
    inertieTotale += inertieInter
    return inertieTotale


def MyKmeans(data, k):
    """
    Implement the Kmeans algorithm
    """
    maxIter = 6
    # Initialize the centroids
    centroids = Initialize_centroids(k, data)
    # Assign the closest centroid to each point
    centroid_assign, centroid_distance = assign_centroid(data, centroids)
    # Compute the inertia
    inertia = ComputeInertieW(data, centroids, centroid_assign)
    print("inertie " + "(0)" + " : " + str(inertia))
    # Display the results
    # init multiple plots
    fig, axs = plt.subplots(int(maxIter/5) +1, 5)
    axs[0, 0].scatter(data[data.columns[0]], data[data.columns[1]], c=centroid_assign)

    # For the number of iterations
    for i in range(1, maxIter):
        # Update the centroids
        centroids = update_centroids(data, centroid_assign, k)
        # Assign the closest centroid to each point
        centroid_assign, centroid_distance = assign_centroid(data, centroids)
        # Compute the inertia
        inertia = ComputeInertieW(data, centroids, centroid_assign)
        print("inertie ", i, " : ", inertia)
        # Add plot
        n = int(i / 5)
        p = i % 5
        axs[n, p].scatter(data[data.columns[0]], data[data.columns[1]], c=centroid_assign)
    plt.show()


MyKmeans(dfA, 3)
