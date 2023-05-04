# The goal of the part 2 is to implement the Kmeans from scratch.

import seaborn as sns
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
#print(dfB)

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
        print ("cluster :", np.array(cluster))
        centroids.append(np.array(cluster).mean(axis=0))
        print("column name :", centroids)
        print("column name :", data.columns[0])
        print("column name :", data.columns[1])
    return pd.DataFrame(centroids,columns = data.columns)

fig, axs = plt.subplots(2, 2)
plt.subplots(2, 2)
# Kmeans first iteration
centroidsA = Initialize_centroids(3, dfA)
centroidsB = Initialize_centroids(3, dfB)

# Assign the closest centroid to each point
centroid_assignA, centroid_distanceA = assign_centroid(dfA, centroidsA)
centroid_assignB, centroid_distanceB = assign_centroid(dfB, centroidsB)

# Display the results in a plot
axs[0,0].scatter(dfA['ApplicantIncome'], dfA['LoanAmount'], c=centroid_assignA)
axs[0,1].scatter(dfB['x'], dfB['y'], c=centroid_assignB)

# Update the centroids
centroidsA = update_centroids(dfA,centroid_assignA, 3)
centroidsB = update_centroids(dfB,centroid_assignB, 3)
centroid_assignA, centroid_distanceA = assign_centroid(dfA, centroidsA)
centroid_assignB, centroid_distanceB = assign_centroid(dfB, centroidsB)

axs[1,0].scatter(dfA['ApplicantIncome'], dfA['LoanAmount'], c=centroid_assignA)
axs[1,1].scatter(dfB['x'], dfB['y'], c=centroid_assignB)
plt.show()

def ComputeInertieW(data,centroids,cluster):
    '''
    Compute the intra-cluster inertia
    '''
    # Select the points that belong to the cluster
    cluster = data[cluster]
    # Compute the distance between each point and the centroid
    dist = DistEucl(cluster, centroids)
    # Compute the sum of the distances
    sum = np.sum(dist)
    return sum

def MyKmeans(data, k=3, max_iter=10, tol=0.001, verbose=True) :
    '''
    Implement the Kmeans algorithm
    '''
    # Initialize the centroids
    centroids = Initialize_centroids(k, data)
    # Assign the closest centroid to each point
    centroid_assign, centroid_distance = assign_centroid(data, centroids)
    # Compute the inertia
    inertia = ComputeInertieW(data, centroids, centroid_assign)
    # Initialize the iteration counter
    iter = 0
    # Initialize the difference between the inertia of two consecutive iterations
    diff = tol + 1
    # While the maximum number of iterations is not reached and the difference between the inertia of two consecutive iterations is greater than the tolerance
    while iter < max_iter and diff > tol:
        # Update the centroids
        centroids = update_centroids(data, centroid_assign, k)
        # Assign the closest centroid to each point
        centroid_assign, centroid_distance = assign_centroid(data, centroids)
        # Compute the inertia
        inertia_new = ComputeInertieW(data, centroids, centroid_assign)
        # Compute the difference between the inertia of two consecutive iterations
        diff = inertia - inertia_new
        # Update the inertia
        inertia = inertia_new
        # Update the iteration counter
        iter += 1
        # Display the results
        if verbose:
            print("Iteration : ", iter)
            print("Inertia : ", inertia)
            print("Difference : ", diff)
            print("Centroids : ", centroids)
            print("Centroid assign : ", centroid_assign)
            print("Centroid distance : ", centroid_distance)
            print("")

    return centroids, centroid_assign, centroid_distance

MyKmeans(dfA, k=3, max_iter=10, tol=0.001, verbose=True)