#!/usr/bin/env python

import math
import numpy as np


def gaussel_partial(A):
    '''
    Performs Gauss elimination on a matrix A. A need not be square. The rows
    of A may be linearly dependent. Returns the upper triangularized form of A.
    '''
    m, n = A.shape
    assert m <= n
    p = np.empty((m-1,), dtype=np.int32)
    for k in range(m-1):
        mu = k + np.argmax(abs(A[k:,k]))
        p[k] = mu
        tmp = np.copy(A[k,k:])
        A[k,k:] = A[mu,k:]
        A[mu,k:] = tmp
        #Forward elimination
        if abs(A[k,k]) >= np.finfo(np.float64).eps:
            A[k+1:,k] /= A[k,k]
            A[k+1:, k+1:] -= np.outer(A[k+1:,k], A[k,k+1:])
    return A, p


def gaussel_full(A):
    m, n = A.shape
    assert m <= n
    p = np.empty((m-1,), dtype=np.int32)
    q = np.empty((m-1,), dtype=np.int32)
    for k in range(m-1):
        nr, nc = divmod(np.argmax(abs(A[k:,k:])), n-k)
        mu = k + nr
        lamda = k + nc
        p[k] = mu
        q[k] = lamda
        #Row swap
        tmp = np.copy(A[k,k:])
        A[k,k:] = A[mu,k:]
        A[mu,k:] = tmp
        #Column swap
        tmp = np.copy(A[:,k])
        A[:,k] = A[:,lamda]
        A[:,lamda] = tmp
        #Forward elimination
        if abs(A[k,k]) >= np.finfo(np.float64).eps:
            A[k+1:,k] /= A[k,k]
            A[k+1:, k+1:] -= np.outer(A[k+1:,k], A[k,k+1:])
    return A, p, q


def gaussel_rank(A):
    m, n = A.shape
    assert m <= n
    p = np.empty((m,), dtype=np.int32)
    q = np.empty((m,), dtype=np.int32)
    rank = 0
    for k in range(m):
        nr, nc = divmod(np.argmax(abs(A[k:,k:])), n-k)
        mu = k + nr
        lamda = k + nc
        p[k] = mu
        q[k] = lamda
        #Row swap
        tmp = np.copy(A[k,k:])
        A[k,k:] = A[mu,k:]
        A[mu,k:] = tmp
        #Column swap
        tmp = np.copy(A[:,k])
        A[:,k] = A[:,lamda]
        A[:,lamda] = tmp
        #Forward elimination
        if abs(A[k,k]) >= np.finfo(np.float64).eps:
            rank += 1
            A[k+1:,k] /= A[k,k]
            A[k+1:, k+1:] -= np.outer(A[k+1:,k], A[k,k+1:])
    print('Rank = ', rank)
    return A, p, q, rank


def gauss_solve(b, A, p, q=None):
    m, n = A.shape
    #Reorder the b vector and apply Gauss transformation to it
    for k in range(m-1):
        pk = p[k]
        tmp = b[k]
        b[k] = b[pk]
        b[pk] = tmp
        b[k+1:] -= b[k]*A[k+1:,k]
    #Backward substitution
    for k in range(m-1,-1,-1):
        b[k] = (b[k] - np.dot(A[k,k+1:], b[k+1:]))/A[k,k]
    #Reorder the solution vector in case of full pivoting
    if q is not None:
        for k in range(m-1):
            qk = q[k]
            tmp = b[k]
            b[k] = b[qk]
            b[qk] = tmp
    return b

#----------------------------------------------------------------------

#A = np.array([[2, -1, 1], [-1, 0, 2], [1, 4, -2]], dtype=np.float64)
#A = np.array([[2, -1, 1, 2, 3], [-4, 2, -2, -4, 6], [1, 4, -2, 1, 5]], dtype=np.float64)
A = np.loadtxt('cm.txt')
#m,n = 3,3
#A = np.random.random((m,n))

print('Matrix A')
print(A)
print('Rank(A) = ', np.linalg.matrix_rank(A))
#b = np.random.random((A.shape[0],))
#b = np.array([0, 5, -5], dtype=np.float64)
#print('Vector b')
#print(b)
#
#x = np.linalg.solve(A, b)
#print('Numpy solution')
#print(x)
#print('\n')

A, p, q = gaussel_full(A)
#print(A)
#print('\n')
#print(p)
#print('\n')
print(q)
#y = gauss_solve(b, A, p, q)
#print('Gauss elem solution')
#print(y)
