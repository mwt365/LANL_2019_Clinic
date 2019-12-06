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

    n_components, n_features = 4097, 300


    # samples = 348
    samples = 1750

    D = generate_dictionary(n_features, n_components)
    X = generate_sparse(n_components, samples)

    # print(D.shape)
    # print(X.shape)

    y = np.dot(D, X)

    spec = sp.Spectrogram('GEN3CH_4_009.dig')

    # axes = plt.axes()

    sgram = spec.spectrogram(0, 50e-6)

    spectrogram_matrix = sgram['spectrogram']
    new_mx = spectrogram_matrix

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=1000)

    print(X.shape)
    print(new_mx.shape)

    # print(spectrogram_matrix[0].reshape(348,1))

    omp.fit(X, new_mx)

    print(omp.get_params())

    coef = omp.coef_

    print(coef)

    print(type(coef.nonzero()[0]))

    idx_r, idx_c = coef.nonzero()

    print(len(idx_c) == len(idx_r))

    for i in range(len(idx_c)):
        assert(new_mx[idx_c[i]][idx_r[i]] != 0)

    print(np.max(idx_c))
    print(np.max(idx_r))

    print((idx_c == idx_r).any())

    bp_mx = np.zeros(shape=sgram['spectrogram'].shape)
    spec2 = sgram['spectrogram']

    for i in range(len(idx_c)):
        # print("c: ",idx_c[i])
        # print("r: ",idx_r[i])
        bp_mx[idx_c[i]][idx_r[i]] = spec2[idx_c[i]][idx_r[i]]
    
    print("its working")

    # plt.plot(idx_c)
    # plt.show()

    # idx_r, = coef.nonzero()
    # plt.subplot(4, 1, 2)
    # plt.xlim(0, 512)
    # plt.title("Recovered signal from noise-free measurements")
    # # plt.stem(idx_r, coef[idx_r], use_line_collection=True)
    # plt.stem(idx_r, coef[idx_r], linefmt='-')

    # sp.plot(axes, sgram)
    fig,ax = plt.subplots(1,2,sharey=True)
    # plt.subplot()


    pcm = ax[0].pcolormesh(sgram['t'] * 1e6, sgram['v'], sgram['spectrogram'])
    plt.gcf().colorbar(pcm, ax=ax[0])
    ax[0].set_ylabel('Velocity (m/s)')
    ax[0].set_xlabel('Time ($\mu$s)')
    title = spec.filename.split('/')[-1]
    ax[0].set_title(title.replace("_", "\\_")+"original")
    
    ax[0].set_ylim(top=4000)
    # plt.show()

    pcm1 = ax[1].pcolormesh(sgram['t'] * 1e6, sgram['v'], bp_mx, cmap='gist_stern_r')
    plt.gcf().colorbar(pcm1, ax=ax[1])
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].set_xlabel('Time ($\mu$s)')
    title = spec.filename.split('/')[-1]
    ax[1].set_title(title.replace("_", "\\_")+"basis pursuit")
    ax[1].set_ylim(top=4000)

    plt.show()







    # print(sgram['spectrogram'].shape)
    # print(y.shape)

