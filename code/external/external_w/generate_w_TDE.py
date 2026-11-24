#!/usr/bin/python
from __future__ import print_function
import sys
from math import exp,log
import numpy as np
import scipy.integrate as integ

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

try :
    w_0           = float(sys.argv[1])
    w_1           = float(sys.argv[2])
    z_t           = float(sys.argv[3])
    Delta_z       = float(sys.argv[4])

# Error control, no need to touch
except IndexError :
    raise IndexError("It seems you are calling this script with too few arguments.")
except ValueError :
    raise ValueError("It seems some of the arguments are not correctly formatted. "+
                     "Remember that they must be floating point numbers.")

# 2. The function giving w(a), including the necessary import statements.
#    Inside this function, you can use the parameters named in the previous step.

def w_TDE_exponent(lna):

    deriv = np.gradient(np.log(f_de(lna)), lna)

    return - deriv


def f_de(lna):

    z=1/np.exp(lna) - 1
  
    Wanalytical = 0.5 * ((w_0+ w_1) + (w_1 - w_0) * np.tanh((z-z_t) / Delta_z))

    return (1+z)**(3*(1+Wanalytical))


# 3. Limits for lna and precision:
#    Check that the boundaries are correct for your case.

nsteps   = 5000.
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

lna=np.array(lna)

log_fde_vec = np.log(f_de(lna))
w_expt = w_TDE_exponent(lna)

# Filling the array of w_a
for N, log_fde, expt_wde in zip(lna,log_fde_vec, w_expt):

    print("%.14g %.14g %.14g" % (N, log_fde, expt_wde))

    # # N=lna , y = \int 3(1+w_DE) dlna, -dy/dlna


