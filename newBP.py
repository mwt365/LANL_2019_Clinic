import spectrogram as sp
from numpy import linalg as la
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal

random_state = np.random.RandomState(0)

n_nonzero_coefs = 50

def generate_dictionary(n_feats, n_comps):

    generator = check_random_state(random_state)

    D = generator.randn(n_feats, n_comps)
    D /= np.sqrt(np.sum((D ** 2), axis=0))

    return D


def generate_sparse(n_comps, n_samples):

    generator = check_random_state(random_state)

    # generate code
    X = np.zeros((n_comps, n_samples))
    for i in range(n_samples):
        idx = np.arange(n_comps)
        generator.shuffle(idx)
        idx = idx[:n_nonzero_coefs]
        X[idx, i] = generator.randn(n_nonzero_coefs)

    return X


if __name__ == "__main__":

    # comps, feats = 348, 512
    # n_nonzero_coefs = 17

    n_components, n_features = 512, 100
    n_nonzero_coefs = 17


    # samples = 348
    samples = 1

    D = generate_dictionary(n_features, n_components)
    X = generate_sparse(n_components, samples)

    # print(D.shape)
    print(X.shape)

    y = np.dot(D, X)

    spec = sp.Spectrogram('GEN3CH_4_009.dig')

    # axes = plt.axes()

    sgram = spec.spectrogram(0, 50e-6)

    spectrogram_matrix = sgram['spectrogram']
    new_mx = spectrogram_matrix[0].reshape(348,1)

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=50)

    # print(X.shape)
    # print(spectrogram_matrix[0].shape)

    # print(spectrogram_matrix[0].reshape(348,1))

    omp.fit(X, new_mx)
    coef = omp.coef_

    # print(coef)

    idx_r, = coef.nonzero()
    # plt.subplot(4, 1, 2)
    # plt.xlim(0, 512)
    # plt.title("Recovered signal from noise-free measurements")
    # # plt.stem(idx_r, coef[idx_r], use_line_collection=True)
    # plt.stem(idx_r, coef[idx_r], linefmt='-')



    # print(sgram['spectrogram'].shape)
    # print(y.shape)

