"""
Cobaya theory codes for a flat universe with a dynamical dark-energy (DE) component.

The expansion rate is

    H(z)^2 = H0^2 [ Omega_m (1+z)^3 + (1 - Omega_m) f_DE(z) ],   f_DE(0) = 1,

where f_DE(z) = rho_DE(z) / rho_DE(0) is the DE density normalised to its value today.
Subclasses only have to define f_DE(z):

- `DDE`                 f_DE = 1 (cosmological constant)
- `ParametricDDE`       w(z) = w0 + wa z/(1+z)
- `GaussianProcessDDE`  f_DE(z) drawn from a Gaussian process (GP)

Provided to the likelihoods
- `luminosity_distance_dde`   : d_L(z), used by the GW likelihood
- `angular_diameter_distance` : d_A(z) = d_L/(1+z)^2, used by the SN likelihoods.

`angular_diameter_distance` is also provided by classy, but classy only knows LCDM here. Any
run with supernovae must therefore set `provides: ['angular_diameter_distance']` in the
block of this theory, otherwise the SNe would be compared with LCDM and not with f_DE(z).
"""

import george
import numpy as np
from george import kernels
from scipy.interpolate import CubicSpline

from cobaya.theory import Theory

C_KM_S = 299792.458


def fde_gp(z, sigma, length, seed, mean=1.0):
    """
    Draw one realisation of f_DE(z) from a GP conditioned on f_DE(0) = 1.

    Squared-exponential kernel of amplitude `sigma` and correlation length `length` (in
    redshift) around a constant `mean`. A given (sigma, length, seed) always returns the
    same curve. george only draws from numpy's global generator, so its state is saved and
    restored: seeding here must not affect anybody else's random numbers.
    """
    kernel = sigma**2 * kernels.ExpSquaredKernel(length**2)
    gp = george.GP(kernel, mean=mean)
    gp.compute(np.array([0.0]), np.array([1e-12]))

    state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(int(seed))
        return gp.sample_conditional(np.array([1.0]), z)
    finally:
        np.random.set_state(state)


class DDE(Theory):
    zmin = 0.0
    zmax = 3.0
    nz_points = 100

    # parameters entering f_DE, to be declared by the subclasses
    fde_params = ()

    def initialize(self):
        self.zs = np.linspace(self.zmin, self.zmax, self.nz_points)

    def get_requirements(self):
        """Every parameter read in `calculate`: cobaya uses this list to know when the
        cached distances are out of date."""
        return {p: None for p in ("h", "omega_b", "Omega_cdm", *self.fde_params)}

    def get_can_provide(self):
        return ["luminosity_distance_dde", "angular_diameter_distance"]

    def get_can_provide_params(self):
        return ["Omega_m", "Omega_Lambda"]

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        for name in self.get_can_provide():
            z = (requirements.get(name) or {}).get("z")
            if z is not None and np.max(z) > self.zmax:
                raise ValueError(
                    f"{name} requested up to z={np.max(z):.3f}, but zmax={self.zmax}")
        return {}

    def fDE(self, z, **params):
        return np.ones_like(z)

    def calculate(self, state, want_derived=True, **params):
        h = params["h"]
        Omega_m = params["omega_b"] / h**2 + params["Omega_cdm"]
        fde = self.fDE(self.zs, **params)

        E2 = Omega_m * (1 + self.zs) ** 3 + (1 - Omega_m) * fde
        # A negative DE density is allowed, a negative H^2 is not
        if not np.all(np.isfinite(E2)) or np.any(E2 <= 0):
            return False  # cobaya reads this as zero likelihood

        dchi_dz = CubicSpline(self.zs, C_KM_S / (100 * h * np.sqrt(E2)))
        state["comoving_distance"] = dchi_dz.antiderivative()
        if want_derived:
            # Omega_Lambda is the DE fraction today, whatever f_DE is
            state["derived"] = {"Omega_m": Omega_m, "Omega_Lambda": 1 - Omega_m}

    def get_luminosity_distance_dde(self, z):
        return (1 + z) * self.current_state["comoving_distance"](z)

    def get_angular_diameter_distance(self, z):
        return self.current_state["comoving_distance"](z) / (1 + z)


class ParametricDDE(DDE):
    fde_params = ("w0_fld", "wa_fld")

    def fDE(self, z, **params):
        w0, wa = params["w0_fld"], params["wa_fld"]
        return (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))


class GaussianProcessDDE(DDE):
    fde_params = ("sigma_f", "ell_f", "seed")

    def get_can_support_params(self):
        # re-parametrisations that are only there to be stored in the chain
        return ["log_sigma_f", "log_ell_f"]

    def fDE(self, z, **params):
        return fde_gp(z, params["sigma_f"], params["ell_f"], params["seed"])
