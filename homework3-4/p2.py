import numpy as np
import matplotlib.pyplot as plt

def find_y(x):
	y = np.zeros(len(x))
	for n in range(1,len(x)-1):
		y[n] = x[n-1] + x[n+1] -2*x[n]
	return y


def plot_x_y(x, y, I):
	plt.figure()
	plt.subplot(2,1,1)
	plt.ylabel('x')
	plt.stem(I,x)
	plt.axis('equal')

	plt.subplot(2,1,2)
	plt.stem(I,y)
	plt.ylabel('Lx')
	plt.axis('equal')


if __name__ == '__main__':
	I = np.linspace(-4, 4, 9, dtype=np.int32)

	x1 = 2*np.ones(9)
	y1 = find_y(x1)
	plot_x_y(x1, y1, I)

	x2 = np.zeros(9)
	x2[4] = 1
	y2 = find_y(x2)
	plot_x_y(x2, y2, I)

	x3 = np.zeros(9)
	x3[4:] = 1
	y3 = find_y(x3)
	plot_x_y(x3, y3, I)

	plt.show()