import numpy as np

def probabilistic_Wiener(L, alpha):
    Rx
    Rxd
    return w_opt


def gen_x(alpha, N):
    x = np.zeros(N)
    s_now, s_prev = np.random.randn(), np.random.randn() # s[0], s[-1]
    x[0] = s_now - 0.5*s_prev # First iteration assumes x[-1]=0
    for n in ?:
        x_prev, s_now = s_now, np.random.randn() # Promote the noise
        x[n] = ?
    return x


def statistical_Wiener(L,x):
    Rx, Rxd = np.zeros((L,L)), np.zeros(L)
    # write your code here
    return w


def LMS(x, L, mu):
    # Fill in the blank
    return w, x_ext


if __name__ == '__main__':
    print 'hello'
