import numpy as np
import matplotlib.pyplot as plt


def convertArray(array):
    return np.array(array)


def displayArray(array):
    print("number of lines: ", array.shape[0])
    if len(array.shape) > 1:
        print("number of columns: ", array.shape[1])
    print("Type of data: ", array.dtype)
    print("size of array: ", array.size)
    print("dimension: ", array.ndim)
    print("-----------------")


def ex1_2_3():
    displayArray(convertArray([1, 2, 3, 4, 5]))
    displayArray(convertArray([[1.5, 2, 3], [4, 5, 6]]))
    boolArray = [[True, True], [False, False]]
    displayArray(convertArray(boolArray))
    zerosArray = np.zeros((3, 4))
    displayArray(zerosArray)
    onesArray = np.ones((3, 4))
    displayArray(onesArray)
    identityArray = np.eye(5)
    displayArray(identityArray)
    emptyArray = np.empty((3, 4))
    displayArray(emptyArray)
    randomArray = np.random.rand(3, 5)
    displayArray(randomArray)
    randomNArray = np.random.randn(3, 5)
    displayArray(randomNArray)
    T = np.linspace(0, 35, 12)
    displayArray(T)
    Treshaped = np.reshape(T, (4, 3))
    displayArray(Treshaped)


def ex4():
    L1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19]
    L1 = convertArray(L1)
    print(L1[0:6])
    print(L1[0:12:2])
    a = np.copy(L1[0:2])
    print(a)
    a[0] = 3
    print(L1)


def ex5():
    T = np.random.rand(3, 5)
    displayArray(T)
    print("+++++")
    print(T)
    print("+++++")
    Tpair = T[:, 0:6:2]
    # print(Tpair)
    Tl1 = T[0]
    # print(Tl1)
    Tc1 = T[:, 0]
    # print(Tc1)
    # print("------")
    X = T[:, 0:-1]
    print(X)
    Y = T[:, -1]
    print(Y)


def dataVisu():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, 'o')
    plt.ylabel('fonction sinus')
    plt.xlabel("l'axe des abcisses")
    plt.show()
