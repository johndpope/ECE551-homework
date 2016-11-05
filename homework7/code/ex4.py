import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    phi = lambda t: (1 - np.abs(t)) * (np.abs(t) < 1)

    N = 5
    t = np.linspace(-N-1, N+1, 500)
    s0 = np.zeros(len(t))
    s1 = np.zeros(len(t))


    plt.figure()

    plt.subplot(2,2,1)
    for n in range(-N, N+1):
        foo = phi(t-n)
        s0 += foo
        plt.plot(t, foo, 'b-')
    plt.axis('equal')
    plt.ylabel('parts of s0')

    plt.subplot(2,2,3)
    plt.plot(t, s0)
    plt.axis('equal')
    plt.ylabel('s0')
 

    plt.subplot(2,2,2)
    for n in range(-N, N+1):
        foo = n*phi(t-n)
        s1 += foo
        plt.plot(t, foo, 'b-')
    plt.axis('equal')
    plt.ylabel('parts of s1')

    plt.subplot(2,2,4)
    plt.plot(t, s1)
    plt.axis('equal')
    plt.ylabel('s1')

    plt.suptitle('N = %d' % N, size=16)
    
    plt.show()
