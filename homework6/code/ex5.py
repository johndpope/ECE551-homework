import numpy as np
import ipdb, os
import matplotlib.pyplot as plt
from scipy import signal

def a_x(k, alpha):
    if k == 0 and alpha == 0:
        return 1.25
    res = (1.25*alpha**k - 0.5*alpha**(k-1) - 0.5*alpha**(k+1)) / (1-alpha**2)
    return res


def probabilistic_Wiener(L, alpha):
    row = np.zeros(L)
    for n in range(L):
        row[n] = a_x(n, alpha)

    Rx = np.zeros((L,L))
    for n in range(L):
        Rx[n] = np.roll(row,n)
    Rx = np.triu(Rx)
    Rx += Rx.T

    Rxd = np.zeros(L)
    Rxd[:L-1] = row[1:]
    Rxd[L-1] = a_x(L, alpha)

    w_opt = np.linalg.pinv(Rx).dot(Rxd)
    return w_opt


def gen_x(alpha, N):
    x = np.zeros(N)
    s_now, s_prev = np.random.randn(), np.random.randn() # s[0], s[-1]
    x[0] = s_now - 0.5*s_prev # First iteration assumes x[-1]=0
    for n in range(1,N):
        x_prev, s_now = s_now, np.random.randn() # Promote the noise
        x[n] = alpha*x[n-1] + s_now - 0.5*s_prev
    return x


def statistical_Wiener(L,x):
    Rx = np.zeros((L,L))
    Rxd = np.zeros(L)

    N = len(x)
    for n0 in range(L):
        #n0 = 0 # starting point, 0 <= n0 < L-len(x)
        X = x[n0:n0+L]; X = X[::-1]
        d = x[L]
        Rx += np.outer(X, X)
        Rxd += d*X
    Rx /= N-L
    Rxd /= N-L

    w = np.linalg.pinv(Rx).dot(Rxd) # pseudo-inverse for singular matrices
    return w


def LMS(x, L, mu):
    w = np.random.rand(L)
    N = len(x)
    for n in range(L, N-1):
        X = x[n-L:n]
        d = x[n]
        e = d - w.dot(X)
        w_new = w + mu*e*X
        #regularizer = -1e-4*np.linalg.norm(w)**2
        #w_new = w + mu*(e*X + regularizer)
        if np.any(np.isnan(w_new)) or np.any(np.isinf(w_new)):
            print 'inf or nan'
            break
        w = w_new
            
    x_est = signal.convolve(x, w, mode='same')
    return w, x_est


if __name__ == '__main__':
    N = 500
    cases = [(alpha, L) for alpha in [0.0, 0.9] for L in [2,4,7]]
    for alpha,L in cases:
        x = gen_x(alpha, N)
        w_opt = probabilistic_Wiener(L, alpha)
        w_stat = statistical_Wiener(L, x)

        mu = 1e-4
        w_hat, _ = LMS(x, L, mu)

        x = gen_x(alpha, N) # regenerate the signal using the same parameters
        x_prob = signal.convolve(x, w_opt, mode='same') # filter x using w_opt
        x_stat = signal.convolve(x, w_stat, mode='same') # filter x using w_stat
        x_lms  = signal.convolve(x, w_hat, mode='same') # filter x using w_hat

        # plot
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(x, 'k', label='original')
        plt.plot(x_prob, 'r', label='probabilistic')
        plt.plot(x_stat, 'g', label='statistical')
        plt.plot(x_lms, 'b', label='LMS')
        plt.grid('on')
        plt.legend()
        plt.title('alpha=%.1f, L=%d' % (alpha, L))

        plt.subplot(2,1,2)
        plt.plot(x-x_prob, 'r', label='probabilistic error')
        plt.plot(x-x_stat, 'g', label='statistical error')
        plt.plot(x-x_lms, 'b', label='LMS error')
        plt.grid('on')
        plt.legend()

        print 'Estimated Wiener error: %.4f' % np.linalg.norm(w_stat - w_opt)
        print 'Estimated LMS error: %.4f' % np.linalg.norm(w_hat - w_opt)
        print '\n'

    plt.show()
