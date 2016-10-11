import numpy as np
import matplotlib.pyplot as plt
import time, ipdb


pi = np.pi
N = 4096
M = 200
I = np.arange(N)
hat_x = {n:np.random.randn() for n in I}
omegas = np.linspace(0, 2*pi, M)

def dtft_approx(I, hat_x, omegas):
    X = np.zeros(M, dtype=complex)
    for k in range(M):
        w = omegas[k]
        for n in range(N):
            X[k] += hat_x[I[n]] * np.exp(-1j*w*n)
    return X


def eq_dtft_approx(hat_x, n0, M):
    X = np.fft.fft(hat_x.values())
    return X


if __name__ == '__main__':
    print 'Approximate X(w)...'
    start = time.time()
    X_1 = dtft_approx(I, hat_x, omegas)
    duration_1 = time.time() - start
    print 'Duration: %.04f s' % duration_1

    print '\n'

    print 'Approximate X(W) using DFT of hat_x...'
    start = time.time()
    n0=0
    X_2 = eq_dtft_approx(hat_x, n0, M)
    duration_2 = time.time() - start
    print 'Duration: %.04f s' % duration_2

    plt.subplot(3,1,1)
    plt.plot(hat_x.values())
    plt.subplot(3,1,2)
    plt.plot(X_1)
    plt.subplot(3,1,3)
    plt.plot(np.real(X_2))
    plt.show()
