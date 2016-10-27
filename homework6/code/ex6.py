import numpy as np
import sys, ipdb 
from ex5 import LMS


if __name__ == '__main__':
    x = np.loadtxt('data.csv', delimiter=',')

    min_mse = sys.maxsize
    best_w, best_L = 0, 0
    for L in range(1,100):
        mu = 1e-3
        w, x_est = LMS(x, L, mu)
        mse = np.mean((x-x_est)**2)
        if mse < min_mse:
            best_w, best_L = w, L

    print 'The estimated order is: %d' % best_L
    print 'The estimated filter: %d' % best_w
