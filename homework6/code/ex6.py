import numpy as np
from scipy import signal
import sys, ipdb 
import matplotlib.pyplot as plt


def LMS(x, L, mu, lambda_reg):
    ''' Using LMS algorithm with regularizer
    '''
    w = np.random.rand(L)
    N = len(x)
    for n in range(L, N-1):
        X = x[n-L:n]
        d = x[n]
        e = d - w.dot(X)
        
        regularizer = -lambda_reg*np.linalg.norm(w)**2
        w_new = w + mu*(e*X + regularizer)
        if np.any(np.isnan(w_new)) or np.any(np.isinf(w_new)):
            print 'inf or nan'
            break
        w = w_new
            
    x_est = signal.convolve(x, w, mode='same')
    return w, x_est


if __name__ == '__main__':
    # read data
    x = np.loadtxt('data.csv', delimiter=',')

    # setup parameters 
    min_mse = sys.maxsize
    best_w, best_L, best_x_est = 0, 0, 0
    mu = 1e-5
    
    stale_count = 0
    tolerance = 1e-5
    patience = 5
    prev_local_min = sys.maxsize

    # main loop
    for L in range(1, 100):
        local_min = sys.maxsize # min mse among different lambdas of one L
        for foo in range(10):
            lambda_reg = 1**-foo
            w, x_est = LMS(x, L, mu, lambda_reg)
            mse = np.mean((x-x_est)**2)

            print 'L=%d, lambda=1e-%d --> mse=%.4f' % (L, foo, mse)
            if mse < min_mse:
                best_w, best_L, best_x_est = w, L, x_est
                min_mse = mse

            if mse < local_min:
                local_min = mse

        # early stopping
        if local_min - prev_local_min < tolerance:
            stale_count += 1
        else:
            stale_count = 0
        if stale_count >= patience:
            print 'Early stopping...\n'
            break
        print 'Stale count = %d\n' % stale_count

    # print out results
    print 'The estimated order is:', best_L
    print 'The estimated filter:', best_w
    print 'The estimated mse:', min_mse

    # plot
    plt.figure()
    plt.plot(x, 'r', label='original')
    plt.plot(best_x_est, 'b', label='estimated')
    plt.legend()
    plt.grid('on')
    plt.show()


