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


def mse(x, x_hat):
    err = x - x_hat
    return np.mean(err**2)


if __name__ == '__main__':
    N = 2 # downsampling / upsampling parameter
    img = misc.imread('lena.bmp', flatten=True)
    g = np.array([0.5,1.0,0.5])

    # part a
    print 'Part a'
    h_a = np.array([1])
    img_a = recover(img, h_a, g, N)
    mse_a = mse(img, img_a)
    print 'MSE=%.04f\n' % mse_a

    # part b
    print 'Part b'
    h_b = g
    img_b = recover(img, h_b, g, N)
    mse_b = mse(img, img_b)
    print 'MSE=%.04f\n' % mse_b

    # part c
    print 'Part c'
    h_c = np.array([1])
    img_c = recover(img, h_c, g, N)
    mse_c = mse(img, img_c)
    print 'MSE=%.04f\n' % mse_c

    # part d
    print 'Part d'
    C = 1/(3+2*np.sqrt(2))
    b = [2*np.sqrt(C), 2*np.sqrt(C)]
    a = [1, 0, C]
    foo = signal.filtfilt(b, a, img, axis=0)
    bar = signal.filtfilt(b, a, img, axis=1)
    img_d = L(U(D( (foo+bar)/2 , N),N), g)
    mse_d = mse(img, img_d)
    print 'MSE=%.04f\n' % mse_d
    

    # plot
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img, cmap='Greys_r')
    plt.xlabel('original image')

    plt.subplot(2,3,2)
    plt.imshow(img_a, cmap='Greys_r')
    plt.xlabel('(a) MSE=%.04f' % mse_a)

    plt.subplot(2,3,3)
    plt.imshow(img_b, cmap='Greys_r')
    plt.xlabel('(b) MSE=%.04f' % mse_b)

    plt.subplot(2,3,5)
    plt.imshow(img_b, cmap='Greys_r')
    plt.xlabel('(c) MSE=%.04f' % mse_c)

    plt.subplot(2,3,6)
    plt.imshow(img_d, cmap='Greys_r')
    plt.xlabel('(d) MSE=%.04f' % mse_d)


    plt.subplot(2,3,4)
    tmp = L(img, g)
    plt.imshow(tmp, cmap='Greys_r')
    plt.xlabel('filtered with g\nMSE=%.04f' % mse(tmp,img))

    plt.show()
