from numpy import zeros
from numpy.linalg import norm
from numpy.random import rand
import numpy as np
import scipy, cv2
import matplotlib.pyplot as plt
import ipdb


def project_on_space(vect, basis):
    projection = np.zeros(vect.shape)
    for n in range(basis.shape[0]):
        projection += np.inner(vect, basis[n]) * basis[n]
    return projection


def gen_S(n, N):
    S = np.zeros((n,N))
    for d in range(N):
        S[:,d] = np.power(np.mgrid[0:n]+1, d) # p
    return S


def gram_schmidt(V):
    Vo = zeros(V.shape) # Create a matrix of similar dimension
    Vo[0] = V[0]/norm(V[0]) # First vector is same vector

    # algorithm 
    for k in range(1, V.shape[0]):
        # projection of k-th row of V onto the space spanned by 
        # orthogonal basis formed by the first k-1 rows in V
        proj = project_on_space(V[k], Vo[:k])

        # residual (projection error)
        res = V[k] - proj

        # new vector
        Vo[k] = res / norm(res)
    return Vo


def gen_circ_tiles(shape, R):
    V = list()
    H, W = shape
    rs, cs = np.mgrid[0:H, 0:W]

    num_circ = int(round(W*H/(R**2)))
    A = np.array([rand(num_circ)*H, rand(num_circ)*W]).T

    for (r,c) in A:
        mask = (cs-c)**2 + (rs-r)**2 <= R**2
        V.append(1*mask) # The 1* will convert boolean to integer

    return np.array(V) # Convert list of 2D arrays to a 3D array


def project_on_tiles(img, tiles):
    pimg = np.zeros(img.shape)

    for q in tiles: # That iterates over the fist dimension
        ipdb.set_trace()
        pimg = pimg + np.dot(q.ravel(), img.ravel())*q

    return np.clip(pimg, 0, 255) # Hard threshold to 8uint values


def main():
    # part a_____________________________________________________________
    V = np.random.random((5,10))
    Vo = gram_schmidt(V)


    # part b_____________________________________________________________
    #img_rgb = scipy.misc.imread('675692436.jpg') 
    img_rgb = cv2.imread('675692436.jpg', 1)
    img = np.mean(img_rgb, axis=2) # Average RGB colors to a single channel
    x = img[17] # Choose some random line number as your wish

    for i in range(1,6):
        proj = project_on_space(x, gen_S(n=1024,N=i).T)
        #plt.figure(); plt.plot(proj)


    # part c_____________________________________________________________
    H,W = img.shape

    # Generate circular tiles. Smaller R -> better (and slower) approximation
    T = gen_circ_tiles(img.shape, 50)
    T = np.reshape(T,(T.shape[0], W*H)) # Flatten tiles from I to 0 .. W*H-1
    To = gram_schmidt(T) # This is where your code should run
    To = np.reshape(To,(To.shape[0], H, W)) # Re-rectangle tiles back to I
    pimg = project_on_tiles(img,To) # Project image

    # Example of plotting images
    plt.subplot(1,2,1)
    plt.imshow(img, interpolation="nearest", cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(pimg, interpolation="nearest", cmap="gray")
    plt.show()
    
    return


if __name__ == '__main__':
    main()