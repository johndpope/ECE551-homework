import numpy as np
import matplotlib.pyplot as plt
import ipdb


def draw_freq(ax, dist, hw, desc='', start=0.0):
    N=20
    centers= np.zeros(N)
    centers[0] = start
    for n in range(1,N):
        centers[n] = centers[n-1]+dist
    x = []
    y = []
    for n in range(N):
        x1 = centers[n] - hw
        x2 = centers[n]
        x3 = centers[n] + hw
        y1, y2, y3 = 0.0, 1.0, 0.0
        plt.plot([x1,x2], [y1,y2], 'k-')
        plt.plot([x2,x3], [y2,y3], 'k-')

    plt.ylabel(desc)
    plt.grid('on')
    ax.set_xlim([-1,5])
    return




if __name__ == '__main__':
    # part a
    plt.figure()
    ax = plt.subplot(5,1,1); draw_freq(ax, dist=2.0, hw=2.0/3.0)
    ax = plt.subplot(5,1,2); draw_freq(ax, dist=1.0, hw=1.0/3.0, desc='U2', start=-1.0)
    ax = plt.subplot(5,1,3); draw_freq(ax, dist=2.0, hw=1.0/3.0, desc='G')
    ax = plt.subplot(5,1,4); draw_freq(ax, dist=2.0, hw=1.0, desc='D3')
    ax = plt.subplot(5,1,5); draw_freq(ax, dist=2.0/3.0, hw=1.0/3.0, desc='U3', start=-2.0/3.0)

    # part b
    plt.figure()
    ax = plt.subplot(5,1,1); draw_freq(ax, dist=2.0, hw=1.0/2.0)
    ax = plt.subplot(5,1,2); draw_freq(ax, dist=1.0, hw=1.0/4.0, desc='U2', start=-1.0)
    ax = plt.subplot(5,1,3); draw_freq(ax, dist=2.0, hw=1.0/4.0, desc='G')
    ax = plt.subplot(5,1,4); draw_freq(ax, dist=2.0, hw=3.0/4.0, desc='D3')
    ax = plt.subplot(5,1,5); draw_freq(ax, dist=2.0/3.0, hw=1.0/4.0, desc='U3', start=-2.0/3.0)
    plt.show()
