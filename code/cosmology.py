import os
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicSpline

from classy import Class

C_KM_S = 299792.458
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FlatBackground:
    """
    Flat background with matter, radiation and a dark energy (DE) whose density
    f_DE(z) = rho_DE(z)/rho_DE(0) is a given function of redshift.

    Same method names and units as classy: distances in Mpc, Hubble in 1/Mpc. Radiation is
    the one of the CLASS defaults (T_cmb = 2.7255 K, 3.044 massless neutrinos).
    """

    def __init__(self, H0, Omega_m, fde, zmax=6.0, nz=3000):
        Omega_r = 2.47298e-5 * (1 + 3.044 * 7 / 8 * (4 / 11)**(4 / 3)) / (H0 / 100)**2
        z = np.linspace(0, zmax, nz)
        E2 = Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + (1 - Omega_m - Omega_r) * fde(z)
        H_over_c = H0 / C_KM_S * np.sqrt(E2)
        self._H = CubicSpline(z, H_over_c, extrapolate=False)
        self._chi = CubicSpline(z, 1 / H_over_c, extrapolate=False).antiderivative()

    def Hubble(self, z):
        return float(self._H(z))

    def luminosity_distance(self, z):
        return float((1 + z) * self._chi(z))

    def angular_distance(self, z):
        return float(self._chi(z) / (1 + z))


def fde_external_w(script, args=()):
    """
    f_DE(z) from a script of code/external/external_w.

    The scripts print three columns: ln(a), ln(f_DE), 3(1+w). They were written for the
    `external_w` option of a patched CLASS, but the background only needs f_DE, so the table
    is read directly and any CLASS can be used. `script` is a path relative to the repo root,
    `args` are the model parameters in the order in which the script reads them.
    """
    path = os.path.join(REPO_ROOT, script)
    out = subprocess.run([sys.executable, path, *map(str, args)],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f'{path} failed:\n{out.stderr}')
    lna, lnf, _ = np.loadtxt(out.stdout.splitlines(), unpack=True)
    recent = lna > np.log(1 / 8)      # the pipeline needs z < 6
    lnf = CubicSpline(lna[recent], lnf[recent])
    return lambda z: np.exp(lnf(-np.log1p(z)) - lnf(0))


# Parameters of each external_w model, in the order in which its script reads them
EXTERNAL_W_ARGS = {
    'CPL': ('w0', 'wa'),
    'CPL*': ('w0', 'wa'),
    'DESI': ('w0', 'wa'),
    'TDE': ('w0', 'w1', 'zt', 'Delta_z'),
    'PEDE': (),
    'Kink': ('w0', 'wm', 'Delta', 'at'),
}


def external_background(cosmology, parameters, settings):
    """Background of a model of EXTERNAL_W_ARGS, with settings = {'script': path}."""
    args = [parameters[p] for p in EXTERNAL_W_ARGS[cosmology]]
    fde = fde_external_w(settings['script'], args)
    return FlatBackground(parameters['H0'], parameters['Omega_m'], fde)


def SetCosmology(cosmology, parameters, type='CLASS', settings=None):

    # 'external' with settings = {'script': ...}: f_DE(z) read from the external_w script, any
    # CLASS. With settings = {'command_w': ...}: below, only with the patched external_w CLASS.
    if type == 'external' and 'script' in (settings or {}):
        return external_background(cosmology, parameters, settings)

    if cosmology == 'LCDM':

        if type == 'CLASS':

            cosmo = Class()
            cosmo.set()
            cosmo.compute()
            return cosmo
            
        elif type == 'external':
            raise ValueError('No external way for LCDM')

        else:
            raise ValueError(
                'You can choose to analyze the chosen cosmology by "CLASS" or "external" only')

    if cosmology == 'CPL':

        if type == 'CLASS':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'w0_fld': parameters['w0'], 'wa_fld': parameters['wa']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        elif type == 'external':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'argw1': parameters['w0'], 'argw2': parameters['wa']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        else:
            raise ValueError('You can choose to analyze the chosen cosmology by "CLASS" or "external" only')

    if cosmology == 'CPL*':

        if type == 'CLASS':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'w0_fld': parameters['w0'], 'wa_fld': parameters['wa']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        elif type == 'external':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'argw1': parameters['w0'], 'argw2': parameters['wa']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        else:
            raise ValueError('You can choose to analyze the chosen cosmology by "CLASS" or "external" only')
        
    if cosmology == 'DESI':

        if type == 'CLASS':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'w0_fld': parameters['w0'], 'wa_fld': parameters['wa']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        elif type == 'external':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'argw1': parameters['w0'], 'argw2': parameters['wa']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        else:
            raise ValueError('You can choose to analyze the chosen cosmology by "CLASS" or "external" only')

    # Modified gravity non-minimally coupled to a scalar field: ONLY WITH hi_class.
    if cosmology == 'non-minimal':

        if type == 'CLASS':

            cosmo = Class()
            cosmo.set(parameters)
            cosmo.set(settings)
            try:
                cosmo.compute()
            except Exception as e:
                raise RuntimeError(
                    "The 'non-minimal' cosmology only works with hi_class, and `classy` must be the "
                    "Python wrapper of hi_class. If you have installed standard CLASS (e.g. `pip install "
                    "classy`) that is why this failed; otherwise see the hi_class error above.") from e
            return cosmo
            
        elif type == 'external':
            raise ValueError('No external way for the non-minimal cosmology')

        else:
            raise ValueError(
                'You can choose to analyze the chosen cosmology by "CLASS" or "external" only')


    if cosmology == 'TDE':

        if type == 'CLASS':

            raise ValueError('Such type of cosmology is not present in CLASS')

        elif type == 'external':

            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'argw1': parameters['w0'], 'argw2': parameters['w1'], 'argw3': parameters['zt'], 'argw4': parameters['Delta_z']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo

        else:
            raise ValueError(
                'You can choose to analyze the chosen cosmology by "CLASS" or "external" only')


    if cosmology == 'PEDE':

        if type == 'CLASS':

            raise ValueError('Such type of cosmology is not present in CLASS')

        elif type == 'external':
            
            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.compute()
            return cosmo

        else:
            raise ValueError(
                'You can choose to analyze the chosen cosmology by "external" only')

    if cosmology == 'Kink':

        if type == 'CLASS':

            raise ValueError('Such type of cosmology is not present in CLASS')

        elif type == 'external':
            
            cosmo = Class()
            cosmo.set(settings)
            cosmo.set({'Omega_m': parameters['Omega_m'], 'H0': parameters['H0']})
            cosmo.set({'argw1': parameters['w0'], 'argw2': parameters['wm'], 'argw3': parameters['Delta'], 'argw4': parameters['at']})
            cosmo.set({'output': 'tCl,pCl,mPk', 'modes': 's', 'lensing': 'no'})
            cosmo.compute()
            return cosmo      

        else:
            raise ValueError(
                'You can choose to analyze the chosen cosmology by "CLASS" or "external" only')



    else:
        raise ValueError('Such type of cosmology is not present in the analysis pipeline')

    