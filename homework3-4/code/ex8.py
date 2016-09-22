import numpy as np
from numpy.random import randint, randn
from scipy.interpolate import splev, splrep
import ipdb


def gen_wave(UIN, n0=0):
    T = np.arange(100)
    Ts = 1+3*np.arange(len(UIN))
    sp = splrep(Ts, UIN, k=3)
    ipdb.set_trace()
    return splev(T-n0, sp, ext=1)


def estimate_delay_and_gain(x1, x2):
    return delta, rho


UIN = np.array([6,7,5,6,9,2,4,3,6])
n1, n2 = randint(1, 40, size=2)
alpha1, alpha2 = randn(2)
x1, x2 = gen_wave(alpha1*UIN, n1), gen_wave(alpha2*UIN, n2)

delta, rho = estimate_delay_and_gain(x1, x2)