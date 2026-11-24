#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
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
    w_m           = float(sys.argv[2])
    Delta         = float(sys.argv[3])
    a_t           = float(sys.argv[4])

# Error control, no need to touch
except IndexError :
    raise IndexError("It seems you are calling this script with too few arguments.")
except ValueError :
    raise ValueError("It seems some of the arguments are not correctly formatted. "+
                     "Remember that they must be floating point numbers.")


# 2. The function giving w(a), including the necessary import statements.
#    Inside this function, you can use the parameters named in the previous step.


def w_Kink_exponent(lna):

    a = np.exp(lna)
    wde = w_0 + (w_m - w_0) * ((1+np.exp(a_t/Delta))/(1+np.exp((a_t-a)/Delta))) * ((1-np.exp((1-a)/Delta))/(1-np.exp(1/Delta)))

    return 3*(1+wde)

def fde_integrand(z):

    wde = w_0 + (w_m - w_0) * ((1+np.exp(a_t/Delta))/(1+np.exp((a_t-(1/(1+z)))/Delta))) * ((1-np.exp((1-(1/(1+z)))/Delta))/(1-np.exp(1/Delta)))
    integrand = wde / (1+z)

    return integrand

def f_de(lna):

    integral = np.zeros_like(lna)
    z=1./np.exp(lna) - 1
    
    for ii in range(len(z)):

        integral[ii] = integ.quad(fde_integrand, 0, z[ii])[0]
    
    return (1+z)**(3+(3/np.log(1+z))*integral)


# 3. Limits for lna and precision:
#    Check that the boundaries are correct for your case.

nsteps    = 5000.
lna_min   = np.log(1.e-15)
lna_max   = np.log(1.)
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
w_expt = w_Kink_exponent(lna)

# Filling the array of w_a
for N, log_fde, expt_wde in zip(lna,log_fde_vec, w_expt):

    print("%.14g %.14g %.14g" % (N, log_fde, expt_wde))

    # # N=lna , y = \int 3(1+w_DE) dlna, -dy/dlna
