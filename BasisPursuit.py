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
    x = np.random.randint(1, size=(1, cols))

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


# def update_element(matrix):

#     row = random.randint(0, len(matrix)-1 )
#     col = random.randint(0, len(matrix[row])-1 )

#     element = matrix[row][col]

#     matrix[row][col] = random.randint(0, element*2)

#     return matrix


def update_dictionary(dictionary):

    index = random.randint(0, len(dictionary)-1 )
    element = dictionary[index]
    dictionary[index] = random.randint(0, int(element*2))

    return dictionary


def update_sparse(sparse):

    # print(sparse[0])

    index = random.randint(0, len(sparse[0])-1 )
    element = sparse[0][index]
    sparse[0][index] = random.randint(0, 1)

    return sparse



def minimize_l2(matrix, epochs, verbose=False):

    min_value, min_dictionary, min_sparse = compute_l2(matrix)
    dictionary = min_dictionary
    sparse = min_sparse

    for i in range(epochs):

        dictionary = update_dictionary(dictionary)
        sparse = update_sparse(sparse)
        dx = np.matmul(dictionary, sparse)

        diff = np.subtract(matrix, dx)
        l2 = calc_l2(diff)

        if min_value > l2:
            min_value = l2
            min_dictionary = dictionary
            min_sparse = sparse

        if verbose:
            print("iteration: ",i)
            print("L2 Norm: ", min_value)
            print("D: ", min_dictionary)
            print("x: ", min_sparse, "\n\n")

    return min_value, min_dictionary, min_sparse



def optimize(matrix, epochs, verbose=False):

    value, dictionary, sparse = minimize_l2(matrix, epochs, verbose)

    matrix2 = np.matmul(dictionary, sparse)

    newMatrix = np.subtract(matrix, matrix2)

    newMatrix[newMatrix < 0] = 0

    return newMatrix




if __name__ == "__main__":

    rows, cols = (5, 5) 
    A = [[random.randint(0,5) for i in range(cols)] for j in range(rows)] 


    newMatrix = optimize(A, 10)
    A = np.array(A)

    newA = np.array(newMatrix)
    print(A, "\n")
    print(newA)

