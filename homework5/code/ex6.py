import numpy as np
import matplotlib.pyplot as plt
import time, ipdb


def dtft_approx(I, hat_x, omegas):
    X_dtft = np.exp(-1j*np.outer(omegas, I)).dot(hat_x)
    X_dtft = np.fft.fftshift(X_dtft)
    return X_dtft


def eq_dtft_approx(hat_x, n0, M):
    X_dft = np.fft.fft(hat_x, M)
    X_dtft = np.fft.fftshift(X_dft)
    return X_dtft


def routine(hat_x, I, M, x_desc=''):
    print '\nM =', M
    omegas = np.linspace(0, 2*pi, M)

    # part a------------------------------------------------
    print 'Approximate X(w)...'
    start = time.time()
    X_1 = dtft_approx(I, hat_x, omegas)
    duration_1 = time.time() - start
    print 'Duration: %.04f s\n' % duration_1

    # part b------------------------------------------------
    print 'Approximate X(W) using DFT of hat_x...'
    start = time.time()
    n0=0
    X_2 = eq_dtft_approx(hat_x, n0, M)
    duration_2 = time.time() - start
    print 'Duration: %.04f s' % duration_2

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(I, hat_x)
    plt.ylabel('hat_x (' + x_desc + ')')

    plt.subplot(3,1,2)
    plt.plot(np.linspace(-np.pi, np.pi, M), X_1)
    plt.ylabel('dtft_approx')

    plt.subplot(3,1,3)
    plt.plot(np.linspace(-np.pi, np.pi, M), X_2)
    plt.ylabel('eq_dtft_approx')
    return 


if __name__ == '__main__':
    pi = np.pi
    N = 4096
    M = 200
    I = np.arange(-N/2, N-N/2)

    hat_x = np.random.random(N)
    routine(hat_x, I, M, 'random')

    hat_x = 5.0*np.ones(N)
    routine(hat_x, I, M, 'constant')

    hat_x = np.zeros(N); hat_x[N/2] = 1
    routine(hat_x, I, M, 'impulse sequence')

    hat_x = np.zeros(N); hat_x[N/2:] = 1
    routine(hat_x, I, M, 'unit step')

    hat_x = np.sqrt(3)*(np.sin(pi*I/3))/(pi*I); hat_x[N/2]=0
    routine(hat_x, I, M, 'sinc')
    
    plt.show()
