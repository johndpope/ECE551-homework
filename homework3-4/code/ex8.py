import numpy as np
from numpy.random import randint, randn
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

def gen_wave(UIN, n0=0):
    T = np.arange(100)
    Ts = 1+3*np.arange(len(UIN))
    sp = splrep(Ts, UIN, k=3)
    return splev(T-n0, sp, ext=1)


def estimate_delay_and_gain(x1, x2):
	c = np.abs(np.correlate(x1, x2, 'same'))
	delta = 50 - np.argmax(c) # the center is 50 since there are 100 elements

	r = np.roll(x1, delta) / x2 # will have nan due to zero division
	rho = r[~np.isnan(r)]
	rho = rho[0]
	return delta, rho


UIN = np.array([6,7,5,6,9,2,4,3,6])
n1, n2 = randint(1, 40, size=2)
alpha1, alpha2 = randn(2)
x1, x2 = gen_wave(alpha1*UIN, n1), gen_wave(alpha2*UIN, n2)

delta, rho = estimate_delay_and_gain(x1, x2)

print 'delta=%d, rho=%.4f' % (delta, rho)
print 'n2=%d, n1=%d, n2-n1=%d' % (n2, n1, n2-n1)
print 'alpha1=%.4f, alpha2=%.4f, alpha1/alpha2=%.4f' % (alpha1, alpha2, alpha1/alpha2)

plt.figure()
plt.subplot(2,2,1); plt.plot(x1); plt.ylabel('x1')
plt.subplot(2,2,2); plt.plot(x2); plt.ylabel('x2')
plt.subplot(2,2,3); plt.plot(np.abs(np.correlate(x1, x2, 'same'))); plt.ylabel('correlation')
plt.show()