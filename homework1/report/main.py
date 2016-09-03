"""
@author: Khoi-Nguyen Mac
"""
import numpy as np
#import ipdb
from index_vector import *


def create_grid_index(N):
	"""Create the stereo 2D integer square grid index set

	Args:
		N: limit for the grid

	Returns:
		I: set of tuples in the form of {(c,i,j)}, where c in {'L','R'} and 0<=i,j<=N-1
	"""
	r = range(N)
	I = {(c,i,j) for c in ['L','R'] for i in r for j in r}
	return I


def inner_product(u, v):
	"""Return the inner product of 2 vectors

	Args:
		u, v: vectors with identical indices

	Returns:
		prod: inner product of u and v
	"""
	assert u.indices == v.indices, 'Vectors have different indices'
	
	prod = 0.0
	for idx in u.indices:
		prod += u[idx] * v[idx]
	return prod


def norm(u):
	"""Return the norm of a vector

	Args:
		u: vector

	Returns:
		norm of u, defined as sqrt(<u,u>), where <.,.> is the inner product
	"""
	return np.sqrt(inner_product(u, u))


def standart_vec(I, t):
	"""Find a standard basis (or reproducing kernel) of a vector

	Args:
		I: index set
		t: arbitrarily chosen index

	Returns:
		et: standard basis, such that u[t] = <u,et>
	"""
	assert t in I, 'Undefined index'
	et = Vector(indices=I, valfun=zeros)
	et[t] = 1.0
	return et


def linear_map(I, inp):
	"""Linear mapping from Vector to list and vice versa

	Args:
		I: index set
		inp: input variable, either a Vector or a list

	Returns:
		out: if inp is a Vector then out is its corresponding list;
			if imp is a list then out is its corresponding Vector
	"""
	if isinstance(inp, Vector):
		out = []
		for i in I:
			out.append(inp[i])
	elif isinstance(inp, list):
		out = Vector(indices=I, valfun=zeros)
		k = 0
		for i in I:
			out[i] = inp[k]
			k += 1
	else:
		print 'Not convertible'
		return
	return out


def conv_matrix(p, L):
	"""Compute the convolution matrix of vector p with a vector of length L

	Args:
		p: array of number
		L: length of the array that p is convolved with

	Returns:
		Tp: convolution matrix (Toeplitz matrix)
	"""
	Tp = np.zeros((len(p)+L-1, L))
	for j in range(L):
		Tp[j:j+len(p),j] = p
	return Tp


def main():
	# part a
	I = create_grid_index(2)
	print 'Grid index:\n', I, '\n'

	# part b
	u = Vector(indices=I, valfun=rand)
	v = Vector(indices=I, valfun=ones)
	print '2*v + u\n', 2*v+u
	print 'u + v\n', u+v
	print '10*u\n', 10*u

	# part c
	print '||u|| = ', norm(u)
	print '||v|| = ', norm(v)
	print '<u,v> = ', inner_product(u,v), '\n'

	# part d
	t = random.sample(I, 1)[0] # randomly choose an index
	et = standart_vec(I, t)
	print 'Standard vector et\n', et
	print 'u[t] - <u, et> =', u[t] - inner_product(u, et), '\n'
	
	# bonus
	u2 = linear_map(I, u)
	u3 = linear_map(I, u2)
	print 'Mapping from Vector to list\n', u2
	print 'Mapping from list to Vector\n', u3

	# part e
	p = [1,-2,1]
	q = [1,2,3,3,2,1]
	L = len(q)
	Tp = conv_matrix(p, L)
	print 'convolution matrix Tp\n', Tp
	print 'Tp*q - convolve(p, q) = ', np.dot(Tp, q) - np.convolve(p, q)
	return


if __name__ == '__main__':
	main()