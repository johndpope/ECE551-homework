import numpy as np
import ipdb
import matplotlib.pyplot as plt
from scipy import signal

def probabilistic_Wiener(L, alpha):
    num = [1,-0.5]; den = [0, -alpha, 1]
    #a_x = signal.filtfilt(num, den

    Rx = 0
    Rxd = 0
    w_opt = 0
    return w_opt


def gen_x(alpha, N):
    x = np.zeros(N)
    s_now, s_prev = np.random.randn(), np.random.randn() # s[0], s[-1]
    x[0] = s_now - 0.5*s_prev # First iteration assumes x[-1]=0
    for n in range(1,N):
        x_prev, s_now = s_now, np.random.randn() # Promote the noise
        x[n] = x[n-1] + s_now - 0.5*s_prev
    return x


def statistical_Wiener(L,x):
    n0 = 0 # starting point, 0 <= n0 < L-len(x)
    X = x[n0:n0+L]; X = X[::-1]
    d = x[L]
    Rx = np.outer(X, X) / L
    Rxd = d*X / L

    w = np.linalg.pinv(Rx).dot(Rxd) # pseudo-inverse for singular matrices
    return w


def LMS(x, L, mu):
    w = np.random.rand(L)
    N = len(x)
    for n in range(L, N-1):
        X = x[n-L:n]
        d = x[n]
        e = d - w.dot(X)
        regularizer = 1e-4*np.linalg.norm(e)**2
        w += mu*(e*X + regularizer)
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            ipdb.set_trace()
            
    ipdb.set_trace()
    x_est = signal.convolve(x, w, mode='same')
    return w, x_est


if __name__ == '__main__':
    N = 500
    cases = [(alpha, L) for alpha in [0, 0.9] for L in [2,4,7]]
    for alpha,L in cases:
        x = gen_x(alpha, N)
        w_opt = probabilistic_Wiener(L, alpha)
        w_stat = statistical_Wiener(L, x)

        #x_prob = # filter x using w_opt
        x_stat = signal.convolve(x, w_stat, mode='same') # filter x using w_stat
        mu = 1e-4
        w_hat, x_lms = LMS(x, L, mu)

        # plot
        plt.figure()
        plt.plot(x, 'k')
        plt.plot(x_stat, 'g')
        plt.plot(x_lms, 'b')
        ipdb.set_trace()
        print 'Estimated Wiener error: %.4f' % np.linalg.norm(w_stat - w_opt)
    plt.show()
