#!/usr/bin/env python

'''
This module provides functions to solve for the roots of a polynomial using
analytical formulas. The formulas are available till only fourth degree
polynomials.
TODO:
    1. Direct method for quartic
    2. Handling of multiple roots via square-free factorization for 
        polynomials. This will require computing the GCD of two polynomials
        in floating point representation.
        See https://en.wikipedia.org/wiki/Root-finding_algorithm 
        and https://en.wikipedia.org/wiki/Square-free_polynomial
        and 
        https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#Euclid.27s_algorithm
'''

import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import brentq, newton
from sympy import mpmath

def print_polynomial(c):
    N = len(c)
    polyaslist = []
    for i in range(N):
        each = c[i]
        if each > 0.0 or each < 0.0:
            if i == 0:
                polyaslist.append('{0}'.format(each))
            else:
                if each > 0.0:
                    polyaslist.append(' + {0} X**{1:d}'.format(each, i))
                else:
                    polyaslist.append(' - {0} X**{1:d}'.format(abs(each), i))

    polyasstr = ''.join(polyaslist)
    print ('\n', polyasstr)
    return None


def wilkinsonpoly(n, kind=1):
    '''
    The Wilkinson polynomials.
    Ref: https://en.wikipedia.org/wiki/Wilkinson%27s_polynomial
    '''
    if kind == 1:
        roots = np.arange(1, (n+1))
    elif kind == 2:
        roots = np.asarray([2**(-i) for i in range(1,(n+1))])
    else:
        raise ValueError('Unknown value for kind')

    c = poly.polyfromroots(roots)
    return c


def quadratic(c, realroots=True):
    '''
    P(x) = c[2]*x**2 + c[1]*x + c[0]
    '''
    #Convert to monic polynomial
    c = c/c[2]

    discriminant = c[1]*c[1] - 4.0*c[0]
    sqrt_discriminant = np.sqrt(abs(discriminant))

    if discriminant < 0.0:
        #Complex roots
        if realroots:
            roots = []
        else:
            root_1 = 0.5*complex(-c[1], sqrt_discriminant)
            root_2 = root_1.conjugate()
            roots = [root_1, root_2]
    elif discriminant > 0.0:
        #Distinct real roots
        #Evaluate the root with greater absolute value first.
        #See Acton, Numerical methods that work.
        if c[1] > 0.0:
            root_1 = 0.5*(-c[1] - sqrt_discriminant)
        else:
            root_1 = 0.5*(-c[1] + sqrt_discriminant)
        root_2 = c[0]/root_1
        roots = [root_1, root_2]
    else:
        #Equal real roots
        root = -0.5*c[1]
        roots = [root, root]
    return roots


def cubic(c, realroots=True):
    '''
    P(x) = c[3]*x**3 + c[2]*x**2 + c[1]*x + c[0]
    Ref: http://www.nickalls.org/dick/papers/maths/cubic1993.pdf
    The Mathematical Gazette (1993); 77 (Nov, No 480), 354-359
    There are typos in the original paper. The link above points to the
    corrected version.
    '''
    #Convert to monic polynomial
    c = c/c[3]

    #Some constants
    third = 1.0/3.0
    sqrt3 = np.sqrt(3.0)
    w1 = complex(-0.5, 0.5*sqrt3)  #First complex cube root of unity
    w2 = w1.conjugate()            #Second complex cube root of unity

    xN = -c[2]/3.0
    yN = c[0] + xN*(c[1] + xN*(c[2] + xN*c[3]))
    deltasq = (c[2]*c[2]-3.0*c[1])/9.0
    hsq = 4.0*deltasq*deltasq*deltasq
    discriminant = yN*yN - hsq
    sqrt_discriminant = np.sqrt(abs(discriminant))

    if discriminant > 0.0:
        #One real root, two complex roots
        p = np.sign(-yN + sqrt_discriminant)*(0.5*abs(-yN + sqrt_discriminant))**third
        q = np.sign(-yN - sqrt_discriminant)*(0.5*abs(-yN - sqrt_discriminant))**third
        root_1 = p + q + xN
        if realroots:
            roots = [root_1]
        else:
            root_2 = p*w1 + q*w2 + xN
            root_3 = root_2.conjugate()
            roots = [root_1, root_2, root_3]
    elif discriminant < 0.0:
        #Three distinct real roots
        pcube = 0.5*complex(-yN, sqrt_discriminant)
        pcube_phase = np.arctan2(pcube.imag, pcube.real)
        pphase = pcube_phase/3.0
        pmodulus = (abs(pcube))**third
        p = complex(pmodulus*np.cos(pphase), pmodulus*np.sin(pphase))
        q = p.conjugate()
        root_1 = (p*w1 + q*w2).real + xN
        root_2 = (p*w2 + q*w1).real + xN
        root_3 = 2.0*p.real + xN
        roots = [root_1, root_2, root_3]
    else:
        #Real roots with multiplicities
        curt_halfyN = np.sign(yN)*abs(0.5*yN)**third
        root_1 = -2.0*curt_halfyN + xN
        root_2 = curt_halfyN + xN
        roots = [root_1, root_2, root_2]
    return roots


def sign_variation(c):
    '''
    Calculates the number of sign variations in a polynomial, neglecting zero
    coefficients.
    '''
    var = 0
    csign = [np.sign(each) for each in c if (each > 0.0 or each < 0.0)]
    N = len(csign)

    for i in range(N-1):
        if np.sign(csign[i]) != np.sign(csign[i+1]):
            var += 1
    return var


def upbound_positive_roots(c, method='Cauchy'):
    '''
    Cauchy bound. The negative of this also gives the lower bound for real
    roots. c[-1] must be non-zero.
    Ref: https://en.wikipedia.org/wiki/Sturm%27s_theorem
    '''
    if method == 'Cauchy':
        ub = 1.0 + (max(np.fabs(c[:-1]))/abs(c[-1]))
    return ub


def sturm_sequence(c):
    '''
    Ref: https://en.wikipedia.org/wiki/Sturm%27s_theorem
    For multiple roots see the two theorems here (but do not use these):
    1. J. M. Thomas, Sturm's theorem for multiple roots, National Mathematics 
        Magazine, Vol. 15, No. 8 (May, 1941) , pp. 391-394
        URL: http://www.jstor.org/stable/3028551
    2. B. M. Meserve, Fundamental concepts of algebra, p. 164, Dover
    '''
    N = len(c)
    seq = []

    if N > 0:
        seq.append(c)
    if N > 1:
        seq.append(poly.polyder(c))
    if N > 2:
        for i in range(2,N):
            try:
                quo, rem = poly.polydiv(seq[i-2], seq[i-1])
            except ZeroDivisionError:
                break
            seq.append(-rem)
    return seq


def eval_sturm_sequence(seq, x):
    '''
    Ref: https://en.wikipedia.org/wiki/Sturm%27s_theorem
    '''
    N = len(seq)
    val = np.zeros((N,))
    val[-1] = seq[-1]
    for i in range(N-1):
        val[i] = poly.polyval(x, seq[i])
    return val


def calc_roots_sturmseq(c):
    '''
    Calculataion of roots based on Sturm Sequence
    '''
    roots = {}
    root_brackets = {}
    all_intervals = []
    lb = 0.0
    ub = upbound_positive_roots(c)
#   print ('Upper bound:', ub)
    seq = sturm_sequence(c)
    varlo = sign_variation(eval_sturm_sequence(seq, lb))
    varhi = sign_variation(eval_sturm_sequence(seq, ub))
    all_intervals.append((lb, ub, varlo, varhi))

    while True:
        #Keep looping over all intervals
        try:
            interval = all_intervals.pop()
        except IndexError:
            break
        lb = interval[0]
        ub = interval[1]
        varlo = interval[2]
        varhi = interval[3]
        numroots = varlo - varhi
        if numroots == 1:
            root_brackets[(lb,ub)] = 1
        elif numroots > 1:
            mid = 0.5*(lb+ub)
            varmid = sign_variation(eval_sturm_sequence(seq, mid))
            all_intervals.append((lb, mid, varlo, varmid))
            all_intervals.append((mid, ub, varmid, varhi))

#   for key in sorted(root_brackets, key=lambda x:x[0]):
#       print (key, root_brackets[key])

    #Polish off the roots
    for bracket, numroots in root_brackets.items():
        sign_lb = np.sign(poly.polyval(bracket[0], c))
        sign_ub = np.sign(poly.polyval(bracket[1], c))
        if (sign_lb*sign_ub < 0.0):
            #Brent's method
            try:
                aroot = brentq(poly.polyval, bracket[0], bracket[1],
                            args=(c,))
            except RuntimeError:
                print ('Not converged')
            roots[aroot] = numroots
        else:
            #Halley's method (Augmented Newton's method with secondderivative as
            #well)
            x0 = 0.5*(bracket[0] + bracket[1])
            aroot = newton(poly.polyval, x0, fprime=eval_polyder, args=(c,),
                    tol=1.48e-08, maxiter=50, fprime2=eval_polyder2)
            roots[aroot] = numroots
    return roots

def eval_polyder(x, c):
    der = poly.polyder(c)
    return poly.polyval(x, der)

def eval_polyder2(x, c):
    der = poly.polyder(c)
    der2 = poly.polyder(der)
    return poly.polyval(x, der2)

def calc_roots(c):
    '''
    Calculataion of roots based on Sympy's arbitrary precision routine.
    Algorithm: Durand-Kerner
    Ref: http://en.wikipedia.org/wiki/Durand-Kerner_method
    '''
    coeffs = c[-1:0:-1]
    #Return all roots, both real and complex. The roots are returned as a sorted
    #list, with the real roots first.
    allroots = mpmath.polyroots(coeffs, maxsteps=50, cleanup=True, extraprec=10,
            error=False)
    #Parse the roots to find only real positive roots
    roots = {}
    realroots = [x for x in allroots if x.imag == 0.0]
    posroots = [x for x in realroots if x > 0.0]
    while posroots:
        aroot = posroots.pop()
        if aroot in roots.keys():
            roots[aroot] += 1
        else:
            roots[aroot] = 1
    for key in roots.keys():
        roots[float(key)] = roots.pop(key)
    return roots

if __name__ == '__main__':
   #Test for quadratic: Real roots
   print ('Test for quadratic: Real roots')
   print ('------------------------------')
   roots = [-1.0, 4.0]
   c = np.zeros((3,))
   c[0] = roots[0]*roots[1]
   c[1] = -(roots[0] + roots[1])
   c[2] = 1.0
   print ('Known Roots: {0}, {1}'.format(*roots))
   print ('Calculated Roots')
   print (quadratic(c))
   print ()

   #Test for quadratic: Complex roots
   print ('Test for quadratic: Complex roots')
   print ('---------------------------------')
   croot = complex(-1.0, 3.0)
   c = np.zeros((3,))
   c[0] = croot.real**2+croot.imag**2
   c[1] = -2.0*croot.real
   c[2] = 1.0
   print ('Known Roots: {0}, {1}'.format(croot, croot.conjugate()))
   print ('Calculated Roots')
   print (quadratic(c, False))
   print ()

   #Test for cubic: Real roots
   print ('Test for cubic: Real roots')
   print ('--------------------------')
   roots = [4.0, -1.0, 1.0]
   c = np.zeros((4,))
   c[0] = -roots[0]*roots[1]*roots[2]
   c[1] = roots[0]*roots[1] + roots[0]*roots[2] + roots[1]*roots[2]
   c[2] = -(roots[0] + roots[1] + roots[2])
   c[3] = 1.0

   print ('Known Roots: {0}, {1}, {2}'.format(*roots))
   print ('Calculated Roots')
   print (cubic(c))
   print ('Roots calculated using numpy routines')
   print (poly.polyroots(c))
   print ()

   #Test for cubic: Complex roots
   print ('Test for cubic: Complex roots')
   print ('-----------------------------')
   rroot = 1.0
   croot = complex(-1.0, 2.0)

   c = np.zeros((4,))
   c[0] = -rroot*(croot.real**2 + croot.imag**2)
   c[1] = rroot*(2.0*croot.real)+ (croot.real**2 + croot.imag**2)
   c[2] = -(rroot + 2.0*croot.real)
   c[3] = 1.0

   print ('Known Roots: {0}, {1}, {2}'.format(rroot, croot, croot.conjugate()))
   print ('Calculated Roots')
   print (cubic(c, False))
   print ('Roots calculated using numpy routines')
   print (poly.polyroots(c))
   print ()

   #Test for sign_variation
   print ('Test for sign variation')
   print ('-----------------------')
   c = np.zeros((4,))
   c[0] = -3.0
   c[1] = 1.0
   c[2] = 0.0
   c[3] = -1.0
   print_polynomial(c)
   print('Sign of coeff vector', np.sign(c))
   print('Number of sign variations (neglecting zeros):', sign_variation(c))

   #Test for sign_variation
   print ('Test for Sturm sequence')
   print ('-----------------------')
   c = np.zeros((21,))
   roots = [1.0, 1.01, 1.02, 1.03, 2.5]
   c = poly.polyfromroots(roots)
   print_polynomial(c)
   print (c)
   seq = sturm_sequence(c)
   for each in seq:
       print_polynomial(each)

   print (upbound_positive_roots(c))
   print ('Roots calculated using numpy routines')
   print (poly.polyroots(c))
   roots = calc_roots(c)
   print('Roots====================')
   for key in sorted(roots):
       print (key, roots[key])

