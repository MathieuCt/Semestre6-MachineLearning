# The goal of the part 2 is to implement the Kmeans from scratch.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.datasets import make_blobs


# read clustering.csv
data = pd.read_csv("clustering.csv")
# create dataframe with "Applic
# antIncome" and "LoanAmount" columns
dfA = data[['ApplicantIncome', 'LoanAmount']]
print(dfA.head())
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
dfB = pd.DataFrame(data2, columns=['x', 'y', 'cluster'])
# Display data frame
print(dfB)

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

# Kmeans first iteration
centroidsA = Initialize_centroids(3, dfA)
#centroidsB = Initialize_centroids(3, dfB)

# Assign the closest centroid to each point
centroid_assignA, centroid_distanceA = assign_centroid(dfA, centroidsA)
#centroid_assignB, centroid_distanceB = assign_centroid(dfB, centroidsB)

# Display the results in a plot
plt.scatter(dfA['ApplicantIncome'], dfA['LoanAmount'], c=centroid_assignA)

plt.show()