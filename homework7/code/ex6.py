import numpy as np
import matplotlib.pyplot as plt
import ipdb


def fwd_bwd_filter(gamma, mu, x):
    v, y = np.zeros(len(x)), np.zeros(len(x))

    v[0] = x[0] # based on your chosen boundary conditions
    for t in range(1, len(x)):
        v[t] = mu*v[t-1] + gamma*x[t]

    y[len(y)-1] = 0 # based on your boundary conditions
    for t in range(len(x)-1, 0, -1):
        y[t-1] = mu*y[t] + gamma*v[t-1]
    return y


def interpolate(t, c, phi):
    s = np.zeros(len(t))
    for n in range(len(c)):
        s += c[n] * phi(t-n)
    return s

if __name__ == '__main__':
    # part b
    phi = [
            lambda t: 1.0*(np.abs(t)<0.5), # phi_0
            lambda t: (1-np.abs(t)) * (np.abs(t)<1), # phi_1
            lambda t: 0.5*(1.5+t)**2 * ((t>=-1.5)*(t<-0.5)) \
                    + (0.75-t**2) * ((t>=-0.5)*(t < 0.5)) \
                    + 0.5*(1.5-t)**2 * ((t>=0.5)*(t<1.5)), # phi_2
            lambda t: np.cos(np.pi*t) * (np.abs(t)<0.5), # phi_3
            lambda t: np.cos(np.pi/2*t) * (np.abs(t)<1) # phi_4
            ]

    filters = [
            lambda x: x,
            lambda x: x, #TODO
            lambda x: fwd_bwd_filter(1.0, 1.0, x), #TODO: change mu, gamma
            lambda x: x, #TODO
            lambda x: x #TODO
            ]
    
    x = [6,7,5,6,9,2,4,3,6]
    N = 500
    t = np.linspace(0,10,N)

    for k in range(len(phi)):
        c = filters[k](x) # compute the coefficients
        s = interpolate(t, c, phi[k]) # compute the interpolating functions here

        plt.subplot(len(phi), 1, k+1)
        plt.plot(t, s)
        plt.plot(np.arange(9), x, 'rx')

    # part c
    N = 5 # number of points
    plt.figure() # open a figure
    plt.axis([0,1,0,1]) # ... and a axis
    points = np.array(plt.ginput(N)) # pick N points using mouse input
    plt.plot(points.T[0], points.T[1], 'rx') # plot them

    t = np.array([])
    for n in range(N-1):
        t = np.concatenate((t, np.linspace(points.T[0][n], points.T[0][n+1], 30)), axis=0)
    c = filters[0](points.T[1])
    s = interpolate(t,c,phi[1])
    plt.figure()
    plt.plot(t, s, 'b-')
    plt.plot(points.T[0], points.T[1], 'rx')

    plt.show()
