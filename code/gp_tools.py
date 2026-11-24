"""
Helper functions of notebooks/mm_gp_analysis.ipynb: the technical steps of the Gaussian
process (GP) analysis, kept out of the notebook to make it easier to read.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cobaya.model import get_model
from cobaya.yaml import yaml_dump, yaml_load_file
from getdist import loadMCSamples
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from theory.DDE import fde_gp

C_KM_S = 299792.458                   # speed of light, km/s
BIN = Path(sys.executable).parent     # where mpirun, cobaya-run and cobaya-install are
CHAINS = "chains/notebook"
Z_GRID = np.linspace(0, 3, 100)       # redshift grid on which theory/DDE.py draws the GP


def set_plot_style():
    """
    Style of the figures of the notebook. The text is written by LaTeX if it is installed,
    otherwise by matplotlib with similar fonts.
    """
    has_latex = all(shutil.which(program) for program in ("latex", "dvipng"))
    plt.rc("text", usetex=has_latex)
    plt.rc("font", size=20, family="sans-serif")
    plt.rc("mathtext", fontset="cm")                     # only used without LaTeX
    plt.rc("axes", titlesize=20, labelsize=24)
    plt.rc("lines", linewidth=1.8, markersize=7)
    plt.rc("legend", frameon=True, framealpha=0.6, fontsize=16)


def download_pantheonplus():
    """Download the Pantheon+ data with Cobaya (~160 MB), if they are not there already."""
    if not Path("packages/data/sn_data/PantheonPlus").exists():
        command = [str(BIN / "cobaya-install"), "sn.pantheonplus", "-p", "packages"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Download of Pantheon+ failed:\n{result.stderr[-2000:]}")


# ---------------------------------------------------------------------------------------
# Distance densities of the events
# ---------------------------------------------------------------------------------------

def mean_and_std(density):
    """Mean and standard deviation of one of the d_L densities (a getdist Density1D)."""
    x = np.linspace(density.x[0], density.x[-1], 4001)
    p = np.clip(density(x), 0, None)
    p = p / p.sum()
    mean = np.sum(x * p)
    std = np.sqrt(np.sum(p * (x - mean)**2))
    return mean, std


# ---------------------------------------------------------------------------------------
# MCMC sampling with Cobaya
# ---------------------------------------------------------------------------------------

def cobaya_settings(run, data_directory, sn_dataset, rminus1_stop, seed):
    """
    Read input_cobaya/<run>.yaml and set which data are used, when the sampling stops, the
    random seed and the output folder.
    """
    info = yaml_load_file(f"input_cobaya/{run}.yaml")
    info["likelihood"]["likelihood.MMcosmologyGP"]["data_directory"] = str(data_directory)
    if "sn.pantheonplus" in info["likelihood"]:
        info["likelihood"]["sn.pantheonplus"]["dataset_file"] = sn_dataset
    info["sampler"]["mcmc"]["Rminus1_stop"] = rminus1_stop
    info["sampler"]["mcmc"]["seed"] = seed
    info["output"] = f"{CHAINS}/{run}/{run}"
    return info


def cobaya_model(info):
    """Cobaya model (theory + likelihoods), to evaluate the likelihood without sampling."""
    return get_model({k: v for k, v in info.items() if k not in ("sampler", "output")})


def run_mcmc(info, n_chains=4, redo=False):
    """
    Sample the posterior, with `n_chains` chains in parallel if mpirun is available. If the
    chains are already there and converged they are kept, unless redo=True.
    """
    output = Path(info["output"])
    checkpoint = output.with_name(output.name + ".checkpoint")
    if checkpoint.exists() and not redo:
        if yaml_load_file(str(checkpoint))["sampler"]["mcmc"]["converged"]:
            print(f"{output.name}: already converged, using the existing chains")
            return

    output.parent.mkdir(parents=True, exist_ok=True)
    settings = output.parent / "settings.yaml"
    settings.write_text(yaml_dump(info))
    command = [str(BIN / "cobaya-run"), str(settings), "--force"]
    if (BIN / "mpirun").exists():
        command = [str(BIN / "mpirun"), "-n", str(n_chains)] + command

    start = time.time()
    log = output.parent / "log.txt"
    # one thread for each chain: multi-threading makes this likelihood ~3x slower
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    with open(log, "w") as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Cobaya failed, see {log}:\n{log.read_text()[-2000:]}")
    print(f"{output.name}: done in {(time.time() - start) / 60:.1f} minutes")


def load_chains(run):
    """Read the chains, without the first 30% (burn-in)."""
    return loadMCSamples(os.path.abspath(f"{CHAINS}/{run}/{run}"),
                         settings={"ignore_rows": 0.3})


def compare_chains(samples):
    """
    Mean of h and Omega_m in each chain, and effective sample size (number of independent
    samples) of Omega_m.
    """
    names = samples.getParamNames().list()
    i_h, i_Om = names.index("h"), names.index("Omega_m")
    rows = []
    for number, chain in enumerate(samples.getSeparateChains(), start=1):
        rows.append({"chain": number,
                     "mean h": np.average(chain.samples[:, i_h], weights=chain.weights),
                     "mean Omega_m": np.average(chain.samples[:, i_Om], weights=chain.weights),
                     "ESS Omega_m": chain.getEffectiveSamples(i_Om)})
    return pd.DataFrame(rows).round({"mean h": 4, "mean Omega_m": 3, "ESS Omega_m": 0})


# ---------------------------------------------------------------------------------------
# From the chains to the curves
# ---------------------------------------------------------------------------------------

def redraw_curves(samples, info, n_curves=4000, seed=43):
    """
    Draw samples from the chain, according to their weights, and compute the curves f_DE(z)
    and H(z) of each one, on Z_GRID.

    The curve of a sample is fixed by (sigma_f, ell_f, seed). The `seed` column is used, not
    int(1e5*random), because `random` is rounded in the chain file.

    Check: the log-likelihood of each redrawn curve must be the one stored in the chain. In a
    few samples (below 1% in our runs) it is not, because the GP is drawn with a matrix decomposition whose
    components have arbitrary signs, and the rounding of sigma_f and ell_f in the chain file
    can flip one of them. These curves are dropped.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(samples.weights), n_curves, p=samples.weights / samples.weights.sum())

    # same model, but taking the seed of the curve as a parameter instead of `random`
    info = dict(info, params=dict(info["params"]))
    del info["params"]["random"]
    info["params"]["seed"] = {"prior": {"min": 0, "max": 5e6}}
    model = cobaya_model(info)
    parameters = list(model.parameterization.sampled_params())

    fde, H, dropped = [], [], 0
    for i in rows:
        point = {p: samples[p][i] for p in parameters}
        loglike = sum(model.loglikes(point, return_derived=False))
        if abs(loglike + samples["chi2"][i] / 2) > 0.01:
            dropped += 1
            continue
        curve = fde_gp(Z_GRID, point["sigma_f"], point["ell_f"], round(point["seed"]))
        Omega_m = samples["Omega_m"][i]
        E2 = Omega_m * (1 + Z_GRID)**3 + (1 - Omega_m) * curve
        fde.append(curve)
        H.append(100 * point["h"] * np.sqrt(E2))

    print(f"{len(fde)} curves redrawn, {dropped} dropped ({dropped / n_curves:.2%})")
    if dropped > 0.05 * n_curves:
        raise RuntimeError("Too many curves do not reproduce the chain")
    return np.array(fde), np.array(H)


def bands(curves, z):
    """
    Median and 68%, 95% bands of the curves (defined on Z_GRID) at the redshifts z.
    Returns the quantiles 2.3%, 15.9%, 50%, 84.1%, 97.7%.
    """
    values = CubicSpline(Z_GRID, curves, axis=1)(z)
    return np.quantile(values, [0.023, 0.159, 0.5, 0.841, 0.977], axis=0)


# ---------------------------------------------------------------------------------------
# H(z) from the slope of the distance-redshift relation
# ---------------------------------------------------------------------------------------

def read_sn(name):
    """
    Mock Pantheon+ supernovae with z > 0.01: table, and covariance matrix of the magnitudes.
    """
    folder = "packages/data/sn_data/PantheonPlus"
    table = pd.read_csv(f"{folder}/Pantheon+SH0ES_{name}.dat", sep=r"\s+")
    numbers = np.loadtxt(f"{folder}/Pantheon+SH0ES_STAT+SYS.cov")   # size, then the elements
    covariance = numbers[1:].reshape(len(table), len(table))
    used = (table.zHD > 0.01).to_numpy()
    return table[used], covariance[np.ix_(used, used)]


def fit_H(knots, z_gw, dL_gw, sigma_gw, sn=None, sn_covariance=None):
    """
    H(z) at the redshifts `knots`, without a model for dark energy.

    c/H(z) is a cubic spline with free values at the knots. Its integral from 0 to z is
    d_L/(1+z) in a flat universe, and it is fitted to the GW distances and, if given, to the
    supernova magnitudes (with a free absolute magnitude M_b).

    Returns H at the knots, its errors and their correlation matrix.
    """
    knots = np.array(knots)
    n = len(knots)
    if sn is not None:
        # this matrix turns correlated SN residuals into independent ones of variance 1
        whiten = np.linalg.inv(np.linalg.cholesky(sn_covariance))

    def residuals(p):
        # p[:n] = 70/H at the knots, p[n] = M_b
        integral = CubicSpline(knots, p[:n] * C_KM_S / 70).antiderivative()

        def comoving_distance(z):
            return integral(z) - integral(0.0)

        r_gw = ((1 + z_gw) * comoving_distance(z_gw) - dL_gw) / sigma_gw
        if sn is None:
            return r_gw
        zhd, zhel = sn.zHD.to_numpy(), sn.zHEL.to_numpy()
        m_b = 5 * np.log10((1 + zhel) * comoving_distance(zhd)) + 25 + p[n]
        r_sn = whiten @ (sn.m_b_corr.to_numpy() - m_b)
        return np.concatenate([r_gw, r_sn])

    start = list(1 / np.sqrt(0.3 * (1 + knots)**3 + 0.7))
    if sn is not None:
        start.append(-19.3)
    fit = least_squares(residuals, start)

    u = fit.x[:n]
    cov_u = np.linalg.inv(fit.jac.T @ fit.jac)[:n, :n]
    sigma_u = np.sqrt(np.diag(cov_u))
    H = 70 / u
    sigma_H = 70 * sigma_u / u**2
    correlation = cov_u / np.outer(sigma_u, sigma_u)
    return H, sigma_H, correlation
