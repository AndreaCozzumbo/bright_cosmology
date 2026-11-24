"""Patches applied to the installed GWFish package on import.

The functions below are modified versions of those of GWFish
(https://github.com/janosch314/GWFish), see THIRD_PARTY_NOTICES.md.

GWFish.modules.fishermatrix.analyze_and_save_to_txt looks up
compute_network_errors and output_to_txt_file as globals in its own module
at call time, so replacing those two attributes on the module object here
is enough for the patched behavior to take effect everywhere, including
inside GWFish's own analyze_and_save_to_txt.

Changes vs. upstream GWFish (modules/fishermatrix.py):
- compute_network_errors: skip signals with zero network SNR instead of
  inverting a degenerate all-zero Fisher matrix.
- output_to_txt_file: format the first output column (the event/population
  name) as a string instead of a float, so event names can be written out.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

import GWFish.modules.fishermatrix as _fm
import GWFish.modules.waveforms as wf
import GWFish.modules.detection as det
from GWFish.modules.fishermatrix import (
    invertSVD,
    sky_localization_area,
    compute_detector_fisher,
)


def compute_network_errors(
    network: det.Network,
    parameter_values: pd.DataFrame,
    fisher_parameters: Optional[list[str]] = None,
    f_ref = wf.DEFAULT_F_REF,
    waveform_model: str = wf.DEFAULT_WAVEFORM_MODEL,
    waveform_class = wf.LALFD_Waveform,
    use_duty_cycle: bool = False,
    redefine_tf_vectors: bool = False,
    save_matrices: bool = False,
    save_matrices_path: Union[Path, str] = Path('.'),
    matrix_naming_postfix: str = '',
    long_wavelength: bool = True,
    eps: float = 1e-5,
    eps_mass: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute Fisher matrix errors for a network whose
    SNR and Fisher matrices have already been calculated.

    Will only return output for the `n_above_thr` signals
    for which the network SNR is above `network.detection_SNR[1]`.

    :param network: detector network to use
    :param parameter_values: dataframe with parameters for one or more signals
    :param fisher_parameters: list of parameters to use for the Fisher matrix analysis - if `None` (default), all waveform parameters are used
    :param waveform_model: waveform model to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param waveform_model: waveform class to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param redefine_tf_vectors: Whether to redefine the time-frequency vectors in order to correctly model signals with small frequency evolution. Defaults to `False`.
    :param use_duty_cycle: Whether to use the detector duty cycle (i.e. stochastically set the SNR to zero some of the time); defaults to `False`
    :param save_matrices: Whether to save the Fisher matrices and their inverses to disk; defaults to `False`
    :param save_matrices_path: Path (expressed with Pathlib or through a string) where  to save the Fisher matrices and their inverses to disk; defaults to `Path('.')` (the current folder)
    :param matrix_naming_postfix: string to be appended to the names of the Fisher matrices and their inverses: they will look like `fisher_matrices_postfix.npy` and `inv_fisher_matrices_postfix.npy`

    :return:
    - `detected`: array with shape `(n_above_thr,)` - array of indices for the detected signals.
    - `network_snr`: array with shape `(n_signals,)` - Network SNR for all signals.
    - `parameter_errors`: array with shape `(n_signals, n_parameters)` - One-sigma     Fisher errors for the parameters.
    - `sky_localization`: array with shape `(n_signals,)` or `None` - One-sigma sky localization area in steradians, returned if the signals have both right ascension and declination, or `None` otherwise.
    """

    if fisher_parameters is None:
        fisher_parameters = list(parameter_values.keys())

    if 'max_frequency_cutoff' in fisher_parameters:
        fisher_parameters.remove('max_frequency_cutoff')

    if 'redshift' in fisher_parameters:
        fisher_parameters.remove('redshift')

    n_params = len(fisher_parameters)
    n_signals = len(parameter_values)

    assert n_params > 0
    assert n_signals > 0

    if isinstance(save_matrices_path, str):
        save_matrices_path = Path(save_matrices_path)

    if save_matrices:
        save_matrices_path.mkdir(parents=True, exist_ok=True)
        fisher_matrices = np.zeros((n_signals, n_params, n_params))
        inv_fisher_matrices = np.zeros((n_signals, n_params, n_params))

    signals_havesky = False
    if ("ra" in fisher_parameters) and ("dec" in fisher_parameters):
        signals_havesky = True
        i_ra = fisher_parameters.index("ra")
        i_dec = fisher_parameters.index("dec")

    detector_snr_thr, network_snr_thr = network.detection_SNR

    parameter_errors = np.zeros((n_signals, n_params))
    if signals_havesky:
        sky_localization = np.zeros((n_signals,))
    network_snr = np.zeros((n_signals,))

    for k in tqdm(range(n_signals)):
        network_fisher_matrix = np.zeros((n_params, n_params))

        network_snr_square = 0.

        signal_parameter_values = parameter_values.iloc[k]

        for detector in network.detectors:

            detector_fisher, detector_snr_square = compute_detector_fisher(detector, signal_parameter_values, fisher_parameters, f_ref, waveform_model,
                                                                           waveform_class, use_duty_cycle, long_wavelength = long_wavelength,
                                                                           eps=eps, eps_mass=eps_mass)

            network_snr_square += detector_snr_square

            if np.sqrt(detector_snr_square) > detector_snr_thr:
                network_fisher_matrix += detector_fisher
        if network_snr_square == 0.:
            continue
        else:
            network_fisher_inverse, _ = invertSVD(network_fisher_matrix)

        if save_matrices:
            fisher_matrices[k, :, :] = network_fisher_matrix
            inv_fisher_matrices[k, :, :] = network_fisher_inverse

        parameter_errors[k, :] = np.sqrt(np.diagonal(network_fisher_inverse))

        network_snr[k] = np.sqrt(network_snr_square)

        if signals_havesky:
            sky_localization[k] = sky_localization_area(
                network_fisher_inverse, parameter_values["dec"].iloc[k], i_ra, i_dec
            )

    detected, = np.where(network_snr > network_snr_thr)

    if save_matrices:

        if matrix_naming_postfix != '':
            if not matrix_naming_postfix.startswith('_'):
                matrix_naming_postfix = f'_{matrix_naming_postfix}'

        fisher_matrices = fisher_matrices[detected, :, :]
        inv_fisher_matrices = inv_fisher_matrices[detected, :, :]

        np.save(save_matrices_path /  f"fisher_matrices{matrix_naming_postfix}.npy", fisher_matrices)
        np.save(save_matrices_path /  f"inv_fisher_matrices{matrix_naming_postfix}.npy", inv_fisher_matrices)

    if signals_havesky:
        return (
            detected,
            network_snr,
            parameter_errors,
            sky_localization,
        )

    return detected, network_snr, parameter_errors, None


def output_to_txt_file(
    parameter_values: pd.DataFrame,
    network_snr: np.ndarray,
    parameter_errors: np.ndarray,
    sky_localization: Optional[np.ndarray],
    fisher_parameters: list[str],
    filename: Union[str, Path],
    decimal_output_format: str = '%.3E'
) -> None:

    if isinstance(filename, str):
        filename = Path(filename)

    delim = " "
    header = (
        "network_SNR "
        + delim.join(parameter_values.keys())
        + " "
        + delim.join(["err_" + x for x in fisher_parameters])
    )
    save_data = np.c_[network_snr, parameter_values, parameter_errors]
    if sky_localization is not None:
        header += " err_sky_location"
        save_data = np.c_[save_data, sky_localization]

    row_format = "%s " + " ".join(["%.3E" for _ in range(parameter_values.shape[1] - 1)]) + " %s " + " ".join(["%.3E" for _ in range(parameter_errors.shape[1])])

    np.savetxt(
        filename.with_suffix(".txt"),
        save_data,
        delimiter=" ",
        header=header,
        comments="",
        fmt=row_format,
    )


_fm.compute_network_errors = compute_network_errors
_fm.output_to_txt_file = output_to_txt_file
