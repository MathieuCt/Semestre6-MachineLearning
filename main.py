import pandas as pd
from sklearn.datasets import load_iris
from Part1 import *


def pandas():
    DF = pd.DataFrame()
    T = np.array([[10, 11], [20, 21]])
    DF2 = pd.DataFrame(T)
    print(DF2.shape)
    #get the name of the dataframe
    print(DF2.columns)
    # 5 first lines
    print(DF2.head())
    # 5 last lines
    print(DF2.tail())
    # number of elements
    print(DF2.size)
    # real data Ndarray
    print(DF2.values)
    Data = [{
        'Name': 'Albert',
        'Email': 'albert@gmail.om',
        'Section': 'SVT'
        }, {
        'Name': 'Nathalie',
        'Email': 'nathalie@gmail.com',
        'Section': 'SC Eco'
        }, {
        'Name': 'Robert',
        'Email': 'robert@gmail.com',
        'Section': 'Math'
        }, {
        'Name': 'Jessika',
        'Email': 'jesika@gmail.com',
        'Section': 'Philo'
        }, {
        'Name': 'Adam',
        'Email': 'adam@gmail.com',
        'Section': 'Physique'
    }]
    DF3 = pd.DataFrame(Data)
    print(DF3.shape)
    # get the name of the dataframe
    print(DF3.columns)
    # 5 first lines
    print(DF3.head())
    # 5 last lines
    print(DF3.tail())
    # number of elements
    print(DF3.size)
    # real data Ndarray
    print(DF3.values)


def pandas3():
    data = [['Tomas', 'tomas@gmail.com', 'Math'],
            ['Albert', 'albert@gmail.com', 'SVT'],
            ['Nathalie', 'nathalie@gmail.com', 'Sc Eco'],
            ['Roberto', 'roberto@gmail.com', 'Physique'],
            ['Adam', 'adam@gmail.com', 'Info']]
    labels = [11, 12, 13, 14, 15]
    # création du DataFrame
    df = pd.DataFrame(data, labels, ['Name', 'Email', 'Section'])
    list_emails = df['Email'].tolist()
    print(list_emails)


def Explore_IrisData():
    # iris = pd.read_csv('iris.csv')
    iris = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(type(iris))
    print(iris.keys())
    print(iris["target_names"])
    print(iris.target_names)
    print(len(iris["data"]))
    print(len(iris["feature_names"]))
    print(np.bincount(iris["target"]))
    print(iris["target"])
    print("data :")
    print(iris["data"])
    plt.hist(iris["data"])
    plt.show()


def part2ex2():
    iris = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # create an hist plot for each feature
    for i in range(4):
        plt.hist(iris["data"][:, i], bins=30)
        plt.title(iris["feature_names"][i])
        plt.show()


def affiche_scatter():
    iris = load_iris()
    for i in range(4):
        for j in range(4):
            plt.scatter(iris["data"][:, i], iris["data"][:, j], c=iris["target"])
            plt.title(iris["feature_names"][i] + " vs " + iris["feature_names"][j])
            plt.show()


if __name__ == '__main__':
    # ex1_2_3()
    # ex4()
    # ex5()
    # dataVisu()
    # pandas1_2()
    # pandas3()
    # Explore_IrisData()
    # part2ex2()
    affiche_scatter()
    



