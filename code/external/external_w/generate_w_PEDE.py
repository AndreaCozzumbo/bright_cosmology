#!/usr/bin/python
from __future__ import print_function
import sys
from math import exp,log
import numpy as np

# README:
#
# This is an example python script for the external_w mode of Class.
# It specifies the Dark Energy behaviour in models beyond LambdaCDM.
# It can be edited and used directly, though keeping a copy of it is recommended.
#
# Two (maybe three) things need to be edited:
#
# 1. The name of the parameters needed for the calculation of the Dark Energy properties.
#    "sys.argv[1]" corresponds to "argw1" in Class, an so on


#PEDE has no free parameters


# 2. The function giving w(a), including the necessary import statements.
#    Inside this function, you can use the parameters named in the previous step.

def w_PEDE(lna) :
    a=exp(lna)
    return -1 - 1./(3*np.log(10))*(1. + np.tanh(-np.log10(a)))

def f_de(lna):
    a=exp(lna)
    return 1 - np.tanh(-np.log10(a))

# 3. Limits for lna and precision:
#    Check that the boundaries are correct for your case.

nsteps=500.
lna_min  = log(1.e-15)
lna_max  = log(1.)
increment = (lna_max-lna_min) / nsteps

#
# And nothing should need to be edited from here on.
#

## Filling the array of lna's
lna = [float(lna_min)]
while lna[-1] <= float(lna_max) :
    lna.append(lna[-1]+increment)

# Filling the array of w_a
for N in lna:
    w = w_PEDE(N)
    # N=lna , y = \int 3(1+w_DE) dlna, dy/dlna
    print("%.14g %.14g %.14g" % (N, log(f_de(N)) , 3*(1+w)))
