#!/usr/bin/python
from __future__ import print_function
import sys
from math import exp,log
import numpy as np
import george
from george import kernels

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

def get_sample_fde(z,sigma_f,ell_f,seed=None):
    '''Compute mean from GP with given training points fde_vals and set of hyperparameters'''
    x_train = np.array([0])
    yerr= np.array([1e-12])
    
    kernel = sigma_f**2 * kernels.ExpSquaredKernel(ell_f)
    
    gp = george.GP(kernel,mean=1)
    gp.compute(x_train, yerr)
    
    y_t=np.array([1])
    
    if seed is not None:
        np.random.seed(int(seed))
        
    pred = gp.sample_conditional(y_t,z)
    
    return pred

def get_wp1times3(lna,lnfde):
    return -np.gradient(lnfde,lna) # d/dln(1+z) = -d/dlna

z_gp=np.linspace(0,10,100)
lna_gp= -np.log(1+z_gp)

lna=np.append(np.log(np.linspace(1e-15,1./(1+z_gp.max()),100,endpoint=False)),lna_gp[::-1])

if __name__=='__main__':
    
    try :
        sigma_f         = float(sys.argv[1])
        ell_f           = float(sys.argv[2])
        seed            = float(sys.argv[3])

    # Error control, no need to touch
    except IndexError :
        raise IndexError("It seems you are calling this script with too few arguments.")
    except ValueError :
        raise ValueError("It seems some of the arguments are not correctly formatted. "+
                        "Remember that they must be floating point numbers.")

    fde=np.zeros_like(lna)
    fde[lna>=lna_gp.min()]=get_sample_fde(z_gp,sigma_f,ell_f,seed)[::-1]
    fde[lna<=lna_gp.min()]=fde[lna==lna_gp.min()]
    fde[fde<=0]=1e-2
    
    lnfde=np.log(fde)
    wp1times3= get_wp1times3(lna,lnfde)

    # Filling the array of w_a
    for N,logfde,wp1t3 in zip(lna,lnfde,wp1times3):
        # Return corresponding arrays N=lna , y = \int 3(1+w_DE) dlna, dy/dlna
        print("%.14g %.14g %.14g" % (N, logfde , wp1t3))
