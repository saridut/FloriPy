#!/usr/bin/env python
'''This module contains routines to perform Gram-Schmidt orthonormalization on
a sequence of vectors.

'''
import numpy as np
import numpy.linalg as la

def gso(A, overwrite=False, out=None):
    '''Performs Gram-Schmidt orthonormalization on a sequence of vectors.

    Parameters
    ----------
    A : ndarray
        (M x N) ndarray with M <= N. The rows of A contain the sequence of
        vectors.
    overwrite : bool, optional
        If `True`, the matrix A is overwritten.
    out : ndarray, optional
        (M x N) ndarray with M <= N. The rows of `out` contain the sequence of
        orthonormal vectors. If `overwrite = True`, `out` is neglected.

    Returns
    -------
    output : ndarray
        (M x N) ndarray with M <= N. The rows of `out` contain the sequence of
        orthonormal vectors. 

    Notes
    -----
    See Golub and Van Loan, Matrix Computations, 3rd edition, Section 5.2.8,
    Algorithm 5.2.5, p. 231.

    '''
    assert A.shape[0] <= A.shape[1]
    M = A.shape[0]
    if overwrite:
        output = A
    else:
        if out is not None:
            output = out
        else:
            output = np.zeros_like(A)
            output[:,:] = A
    for i in range(M):
        output[i,:] = output[i,:]/la.norm(output[i,:])
        for j in range(i+1, M):
            output[j,:] = output[j,:] - np.dot(output[j,:], output[i,:])*output[i,:]
    return output


if __name__ == '__main__':
    A = np.random.random((6,6))
    print('A')
    print(A)
    out = gso(A)
    print('\n')
    print(out)
    print('\n')
    print(np.dot(out.T, out))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            print(i, j, np.dot(out[i,:], out[j,:]))
        print('\n')




