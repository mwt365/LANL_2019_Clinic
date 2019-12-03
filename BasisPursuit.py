import numpy as np
from numpy import linalg as la
import random


# random.seed(12345)
A = [[3, 0, 2, 2, 1],
    [2, 3, 1, 2, 0],
    [3, 2, 1, 1, 2],
    [0, 3, 1, 1, 1],
    [0, 1, 2, 2, 0]]

def pretty_print(matrix):
    for i in range(len(matrix[0])):
        print(matrix[i])

def calc_l2(matrix):
    return la.norm(matrix, 'fro')

def init_dict(matrix):
    rows = len(matrix)
    D = np.random.randint(5, size=(rows, 1))

    return D

def init_sparse(matrix):
    cols = len(matrix[0])
    x = np.random.randint(5, size=(1, cols))

    return x

def calc(matrix1, dictionary, sparse):
    matrix2 = np.matmul(dictionary, sparse)
    difference = np.subtract(matrix1, matrix2)

    return difference

def compute_l2(matrix):
    dictionary = init_dict(matrix)
    sparse = init_sparse(matrix)
    diff_mx = calc(matrix, dictionary, sparse)
    l2 = calc_l2(diff_mx)

    return (l2, dictionary, sparse)


def minimize_l2(matrix, epochs):

    min_l2 = (float('inf'), [], [])

    for i in range(epochs):

        l2 = compute_l2(matrix)

        if min_l2[0] > l2[0]:
            min_l2 = l2

    value = l2[0]
    dictionary = l2[1]
    sparse = l2[2]

    return value, dictionary, sparse


rows, cols = (30, 30) 
A = [[random.randint(0,20) for i in range(cols)] for j in range(rows)] 





value, dictionary, sparse = minimize_l2(A, 100)

# print(dictionary)
# print(sparse)
# print()

matrix2 = np.matmul(dictionary, sparse)
diff = np.subtract(A, matrix2)
print(np.absolute(diff))

A = np.array(A)
print(A)

print(value)