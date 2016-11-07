import numpy as np
import matplotlib.pyplot as plt
import ipdb


def fwd_bwd_filter(gamma, mu, x):
    v, y = np.zeros(len(x)), np.zeros(len(x))

    v[0] = x[0] # based on your chosen boundary conditions
    for t in range(1, len(x)):
        v[t] = mu*v[t-1] + gamma*x[t]

    y[len(y)-1] = v[-1] # based on your boundary conditions
    for t in range(len(x)-1, 0, -1):
        y[t-1] = mu*y[t] + gamma*v[t-1]
    return y


def interpolate(t, c, phi):
    s = np.zeros(len(t))
    for n in range(len(c)):
        s += c[n] * phi(t-n)
    return s


if __name__ == '__main__':
    # part b---------------------------------------------------------------------
    phi = [
            lambda t: 1.0*(np.abs(t)<0.5), # phi_0
            lambda t: (1-np.abs(t)) * (np.abs(t)<1), # phi_1
            lambda t: 0.5*(1.5+t)**2 * ((t>=-1.5)*(t<-0.5)) \
                    + (0.75-t**2) * ((t>=-0.5)*(t < 0.5)) \
                    + 0.5*(1.5-t)**2 * ((t>=0.5)*(t<1.5)), # phi_2
            lambda t: 0.5*(1.0+np.cos(np.pi*t)) * (np.abs(t)<1.0), # phi_3
            lambda t: np.cos(np.pi*t) * (np.abs(t)<0.5) # phi_4
            ]

    mu = np.sqrt(8/(2*np.sqrt(2)+3))
    gamma = 2*np.sqrt(2)-3
    filters = [
            lambda x: x,
            lambda x: x,
            lambda x: fwd_bwd_filter(mu, gamma, x),
            lambda x: x,
            lambda x: x, 
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
        plt.ylabel('phi'+str(k))


    # part c---------------------------------------------------------------------
    N = 5 # number of points
    plt.figure() # open a figure
    plt.axis([0,1,0,1]) # ... and a axis
    plt.grid('on')
    points = np.array(plt.ginput(N)) # pick N points using mouse input
    plt.plot(points.T[0], points.T[1], 'rx') # plot them
    plt.close()
    

    t = np.linspace(0,N-1,500)
    plt.figure()
    for k in range(len(phi)):
        c0 = filters[k](points.T[0])
        c1 = filters[k](points.T[1])
        s0 = interpolate(t, c0, phi[k])
        s1 = interpolate(t, c1, phi[k])

        plt.plot(s0, s1, label='\phi'+str(k))

    plt.plot(points.T[0], points.T[1], 'o')
    plt.axis([0,1,0,1]); plt.grid('on'); plt.legend()

    plt.show()
