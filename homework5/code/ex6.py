import numpy as np
import matplotlib.pyplot as plt
import time, ipdb


def dtft_approx(I, hat_x, omegas):
    X = np.exp(-1j*np.outer(omegas, I)).dot(hat_x)
    X = np.fft.fftshift(X)
    return X


def eq_dtft_approx(hat_x, n0, M):
    N = len(hat_x)
    if M >= N:
        L = M-N # pad to make len(hat_x) = M
    else:
        L = int(np.ceil(N*1.0/M)*M - N) # pad to make len(hat_x) = lowest multiple of M
    x_pad = np.pad(hat_x, (0,L), 'constant', constant_values=0.0)
    X = np.fft.fft(x_pad)

    if len(X) > M:
        X = X[::len(X)/M]

    X = np.fft.fftshift(X)
    return X


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
    X_2 = eq_dtft_approx(hat_x, I[0], M)
    duration_2 = time.time() - start
    print 'Duration: %.04f s' % duration_2

    plt.figure()
    ax = plt.subplot(3,1,1)
    plt.plot(I, hat_x)
    ax.set_xlim([I[0],I[-1]])
    ax.set_ylim([hat_x.min()-0.1, hat_x.max()+0.1])
    plt.ylabel('hat_x (' + x_desc + ')')

    ax = plt.subplot(3,1,2)
    plt.plot(np.linspace(-np.pi, np.pi, M), np.abs(X_1))
    ax.set_xlim([-np.pi, np.pi])
    plt.ylabel('dtft_approx')

    ax = plt.subplot(3,1,3)
    plt.plot(np.linspace(-np.pi, np.pi, M), np.abs(X_2))
    ax.set_xlim([-np.pi, np.pi])
    plt.ylabel('eq_dtft_approx')
    return 


if __name__ == '__main__':
    pi = np.pi
    N = 4096
    M = 5000
    I = np.arange(-N/2, N-N/2)

    hat_x = np.random.random(N)
    routine(hat_x, I, M, 'random')
    
    hat_x = 5.0*np.ones(N)
    routine(hat_x, I, M, 'constant')

    hat_x = np.zeros(N); hat_x[N/2] = 1
    routine(hat_x, I, M, 'impulse sequence')

    hat_x = np.zeros(N); hat_x[N/2:] = 1
    routine(hat_x, I, M, 'unit step')

    hat_x = np.sqrt(3)*(np.sin(pi*I/3))/(pi*I); hat_x[N/2]=hat_x[N/2-1]
    routine(hat_x, I, M, 'sinc')
    
    plt.show()
