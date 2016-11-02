import numpy as np
import matplotlib.pyplot as plt
import ipdb

phi = lambda t: np.maximum(0.0, 1.0 - np.abs(t))

if __name__ == '__main__':
    N = 5
    resolution = 0.01
    t = np.arange(-N-1,N+1,resolution)

    # s0
    s0 = np.zeros(len(t))
    plt.figure()
    plt.subplot(2,1,1)
    for n in range(-N, N+1, 1):
        plt.plot(t, phi(t-n), 'b')
        s0 += phi(t-n)
    plt.ylabel('phi(t-n)'); plt.grid('on'); plt.axis('equal'); plt.title('s0(t), N=%d' % N)

    plt.subplot(2,1,2)
    plt.plot(t,s0, 'b')
    plt.ylabel('s0'); plt.grid('on'); plt.axis('equal')

    # s1
    s1 = np.zeros(len(t))
    plt.figure()
    plt.subplot(2,1,1)
    for n in range(-N, N+1, 1):
        plt.plot(t, n*phi(t-n), 'b')
        s1 += n*phi(t-n)
    plt.ylabel('n*phi(t-n)'); plt.grid('on'); plt.axis('equal'); plt.title('s1(t), N=%d' % N)

    plt.subplot(2,1,2)
    plt.plot(t,s1, 'b')
    plt.ylabel('s1'); plt.grid('on'); plt.axis('equal')

    plt.show()
