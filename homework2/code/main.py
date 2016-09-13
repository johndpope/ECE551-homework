from numpy import zeros
from numpy.linalg import norm
import numpy as np
import scipy
import matplotlib.pyplot as plt
import ipdb


def gram_schmidt(V):
    Vo = zeros(V.shape) # Create a matrix of similar dimension
    Vo[0] = V[0]/norm(V[0]) # First vector is same vector

    # algorithm
    return Vo


def gen_circ_tiles(shape, R):
    V = list()
    H, W = shape
    rs, cs = np.mgrid[0:H, 0:W]

    num_circ = round(W*H/(R**2))
    A = np.array([rand(num_circ)*H, rand(num_circ)*W]).T

    for (r,c) in A:
        mask = (cs-c)**2 + (rs-r)**2 <= R**2
        V.append(1*mask) # The 1* will convert boolean to integer

    return np.array(V) # Convert list of 2D arrays to a 3D array


def project_on_tiles(img, tiles):
    pimg = np.zeros(img.shape)

    for q in tiles: # That iterates over the fist dimension
        pimg = pimg + np.dot(q.ravel(), img.ravel())*q

    return np.clip(pimg, 0, 255) # Hard threshold to 8uint values


def main():
    # part a
    ipdb.set_trace()
    V = np.random.random((5,10))
    Vo = gram_schmidt(V)

    # part b
    img_rgb = scipy.misc.imread('675692436.jpg') # Average RGB colors to a single channel
    img = np.mean(img_rgb, axis=2) # Choose some random line number as your wish
    x = img[17]

    # part c
    
    return


if __name__ == '__main__':
    main()