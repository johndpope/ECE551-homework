import numpy as np
import scipy, scipy.signal
import matplotlib.pyplot as plt
import ipdb


def gen_sin_signal(a, f, t):
    return np.sin(a * 2.0 * np.pi * f * t)


# part a-----------------------------------------------------------------------------
def downsample(x,N):
    y = x[::N]
    return y


def upsample(x,N):
	y = np.zeros(len(x)*N)
	y[::N] = x
	return y


# part b-----------------------------------------------------------------------------
N = 30 # length of filter
wc = np.pi / 3 # cutoff frequency
sample_rate = 100.0
nyq_rate = sample_rate / 2.0

fir = scipy.signal.firwin(N, wc / nyq_rate) # fir low pass
FIR = scipy.signal.freqz(fir) # frequency response of fir

plt.figure()
plt.subplot(2,2,1); plt.stem(FIR[1]); plt.xlabel('Frequency response of FIR')
plt.subplot(2,2,3); plt.stem(downsample(FIR[1],2)); plt.xlabel('Downsampled by 2')
plt.subplot(2,2,4); plt.stem(upsample(FIR[1],2)); plt.xlabel('Upsampled by 2')


# part c-----------------------------------------------------------------------------
samples = 8192
time_interval = 1
t = np.linspace(0, time_interval, samples)

f1, a1 = 800, 1
f2, a2 = 1600, 0.2
f3, a3 = 2400, 0.4
x1 = gen_sin_signal(a1, f1, t)
x2 = gen_sin_signal(a2, f2, t)
x3 = gen_sin_signal(a3, f3, t)
x = x1 + x2 + x3

y1 = downsample(x, 3)
y2 = downsample(scipy.signal.convolve(x, fir), 3)
Y1 = scipy.signal.freqz(y1)
Y2 = scipy.signal.freqz(y2)

plt.figure()
plt.subplot(2,1,1); plt.plot(Y1[1]); plt.xscale('log'); plt.xlabel('y1 = D3 x')
plt.subplot(2,1,2); plt.plot(Y2[1]); plt.xscale('log'); plt.xlabel('y2 = D3 L x')

plt.show()