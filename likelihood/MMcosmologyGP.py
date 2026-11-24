"""
A Cobaya likelihood for bright sirens: gravitational-wave events with a known redshift

Author: Andrea Cozzumbo
Email: andrea.cozzumbo@gssi.it

If you use it, please cite
    Cozzumbo et al., "Model-independent cosmology with joint observations of gravitational
    waves and gamma-ray bursts", JCAP 05 (2025) 021
    https://iopscience.iop.org/article/10.1088/1475-7516/2025/05/021
    https://arxiv.org/abs/2411.02490
"""
import os

import dill as pickle
import numpy as np

from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError


class MMcosmologyGP(Likelihood):
    """
    Likelihood of the luminosity distances d_L of GW events with a known redshift. The data
    are those produced by `code/mm_gw.py`. Two options for `interpolation_method`:

    - 'kde'      : product over the events of the probability densities p(d_L), each
                   evaluated at the d_L that the theory predicts at the redshift of the event.
                   With or without peculiar velocities (`peculiar_velocity`).
    - 'gaussian' : usual chi^2, with the d_L of the events and the errors of the Fisher matrix
                   of GWFish. These errors do not include the priors (e.g. on the angle of the
                   jet) nor the peculiar velocities.
    """

    def initialize(self):
        self.method = str(self.interpolation_method).lower()
        if self.method not in ('kde', 'gaussian'):
            raise LoggedError(self.log, "'interpolation_method' can be 'kde' or 'gaussian', "
                              f"got {self.interpolation_method!r}")
        with_vp = self.peculiar_velocity not in (None, 'None', 'No')
        if self.method == 'gaussian' and with_vp:
            raise LoggedError(self.log, "The errors used by 'interpolation_method: gaussian' do "
                              "not include the peculiar velocities: set 'peculiar_velocity: "
                              "None', or use 'interpolation_method: kde'")

        self.data_file = f"{self.gw_network}_{self.fiducial_cosmology}_{self.grb_dataset}.pkl"
        path = os.path.join(self.data_directory, self.data_file)
        if not os.path.isfile(path):
            raise LoggedError(self.log, f"Could not find {self.data_file} in "
                              f"{os.path.abspath(self.data_directory)}. Please provide the "
                              "directory where the data files are stored")
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        # Layouts written by mm_gw_pipeline.GWFishPosterior:
        #   [events, z, (posteriors,) densities, dL, sigma_dL(, densities_vp)]
        # 'posteriors' is there only for SavePosteriorBool: True, 'densities_vp' only for
        # PeculiarCorrectionBool: True
        # 'densities' is not needed by the gaussian method, and can be None
        has_posteriors = (dataset[2] is not None and len(dataset[2]) > 0
                          and isinstance(dataset[2][0], np.ndarray))
        i = 3 if has_posteriors else 2
        self.event = dataset[0]
        self.z = np.atleast_1d(np.squeeze(dataset[1]))
        self.num_events = len(self.z)

        if self.method == 'gaussian':
            self.log.info('Gaussian likelihood with the errors of the Fisher matrix')
            self.dL = np.array(dataset[i + 1], dtype=float)
            self.sigma_dL = np.array(dataset[i + 2], dtype=float)
        elif not with_vp:
            self.log.info('No peculiar velocity correction applied')
            self.density = dataset[i]
        else:
            if len(dataset) <= i + 3:
                raise LoggedError(self.log, f"{self.data_file} has no peculiar-velocity "
                                  "corrected densities: set 'peculiar_velocity: None'")
            self.log.info('Using the peculiar-velocity corrected densities')
            self.density = dataset[i + 3]

        self.log.info("Read %d events from %s", self.num_events, path)

    def get_requirements(self):
        return {'luminosity_distance_dde': {'z': self.z}}

    def logp(self, **params_values):
        dL_th = self.provider.get_luminosity_distance_dde(self.z)

        if self.method == 'gaussian':
            return -0.5 * np.sum(((dL_th - self.dL) / self.sigma_dL)**2)

        p = np.array([float(density(dL)) for density, dL in zip(self.density, dL_th)])
        # the densities vanish outside the sampled range
        if not np.all(p > 0):
            return -np.inf
        return np.sum(np.log(p))
