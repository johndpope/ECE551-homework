import numpy as np
import scipy
from scipy import misc, signal
import matplotlib.pyplot as plt
import ipdb


def L(x, kernel):
    y = signal.convolve2d(x, np.outer(kernel, kernel), mode='same')
    return y


def D(x, N): # Downsampling by N
    y = x[::N, ::N]
    return y


def U(x, N): # Upsampling by N
    y = np.zeros((x.shape[0]*N, x.shape[1]*N))
    y[::N,::N] = x
    return y


def recover(img, h, g, N):
    img_hat = L(U(D(L(img, h), N), N), g)
    return img_hat


if __name__ == '__main__':
    M = 10 # filter length
    N = 2 # downsampling / upsampling parameter
    img = misc.imread('lena.bmp', flatten=True)
    g = np.ones(M); g[0] = 0.5; g[-1] = 0.5

    # part a
    h_a = np.array([1])
    img_a = recover(img, h_a, g, N)

    # part b
    h_b = g
    img_b = recover(img, h_b, g, N)

    # part c
    h_c = np.array([1])
    img_c = recover(img, h_c, g, N)

    # part d
    C = 1/(3+2*np.sqrt(2))
    #foo = signal.filtfilt([0,2,4,2], [1,0,6,0,1], img)
    foo = signal.filtfilt([2*np.sqrt(C), 2*np.sqrt(C)], [C,0,1], img)
    ipdb.set_trace()
    img_d = L(U(D(foo, N),N), g)
    

    # plot
    plt.figure()
    plt.subplot(2,2,1); plt.imshow(img_a, cmap='Greys_r')
    plt.subplot(2,2,2); plt.imshow(img_b, cmap='Greys_r')
    plt.subplot(2,2,3); plt.imshow(img_b, cmap='Greys_r')
    plt.subplot(2,2,4); plt.imshow(img_d, cmap='Greys_r')

    plt.show()
