#!/usr/bin/env python

import math
import numpy as np

def decomp_LTL(A, overwrite=False, zero_upper=True, out=None):
    m, n = A.shape
    assert m == n
    if overwrite:
        L = A
    else:
        if out is not None:
            assert n,n == out.shape
            out[:,:] = 0.0
            L = out
        else:
            L = np.zeros((n,n))

    for i in range(n-1,-1,-1):
        L[i,i] = math.sqrt(A[i,i] - np.dot(L[i+1:,i],L[i+1:,i]))
        for j in range(i):
            L[i,j] = (A[i,j] - np.dot(L[i+1:,i], L[i+1:,j]))/L[i,i]

    if overwrite and zero_upper:
        for i in range(n):
            for j in range(i+1,n):
                L[i,j] = 0.0
    return L

np.set_printoptions(linewidth=160)
n = 6
R = np.random.random((n,n))
A = np.dot(R, R.T)
B = np.copy(A)

L = decomp_LTL(B, overwrite=True)

print('A---------------')
print(A)
print('L---------------')
print(L)

print(np.allclose(A, np.dot(L.T, L)))
