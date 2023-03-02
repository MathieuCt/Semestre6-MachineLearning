import pandas as pd
from sklearn.datasets import load_iris
from Part1 import *



def padas1_2():
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


def part2ex2():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))


    }
    '''
    # df = iris
    print(df.head())
    # type of variable
    print(df.dtypes)
    # print available columns information
    print("-----------------")
    print(df.keys())
    print("-----------------")
    print(iris['target_names'])
    print("-----------------")
    print(iris.target_names)
    print("-----------------")
    # print the number of lines in a dataset
    print(iris['data'].shape[0])
    print("-----------------")
    # print the number of columns in a dataset
    print(iris['data'].shape[1])
    print("-----------------")
    # Name of the columns
    print(iris['feature_names'])
    print("-----------------")
    # Les labels de chaque classe sont donnés par l’attribut "target". Afficher la taille de
    # chaque dimension de l’attribut target. Afficher les labels des différentes observations.
    print(iris['target'].shape)
'''


if __name__ == '__main__':
    #ex1_2_3()
    # ex4()
    # ex5()
    # dataVisu()
    # pandas1_2()
    # pandas3()
    # Explore_IrisData()
    part2ex2()



