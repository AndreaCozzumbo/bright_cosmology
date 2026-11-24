"""
03/07/2023
A python script by Andrea Cozzumbo, Ulyana Dupletsa, Riccardo Murgia
devised to produce {z, d_l, Deltad_l} mock data from ET + high-energy satellites (Fermi-like)
theoretical d_l are computed by class or hi-class
observerd d_l and uncertanties are computed via gwfish

"""

import argparse
import os

import read_input

parser = argparse.ArgumentParser(description="GW mock data: catalog -> GWFish -> dL posteriors")
parser.add_argument("--config", default=read_input.CONFIG_FILE,
                    help="yaml config file (default: code/inputs.yaml)")
read_input.CONFIG_FILE = os.path.abspath(parser.parse_args().config)
# the paths in the config file are relative to the repo root
os.chdir(read_input.REPO_ROOT)

import mm_gw_pipeline
#import mm_gw_pipeline_gaussian as mm_gw_pipeline
from classy import Class  ###if you want to produce mock data in MG models, ensure you are importing hi-class
import cosmology
import numpy as np
from scipy.interpolate import interp1d
############################### INPUT ##########################################

input_dct = mm_gw_pipeline.input_dct

################################################################################
###################### your favourite DE/MG models #############################

cosmo_name          = input_dct['CosmologicalModel']

params              = input_dct['Parameters']

cosmo_analysis_type = input_dct['Mode']

settings            = input_dct['Settings']

MG_bool             = input_dct['MGBool']

MG_params           = input_dct['MGparameters']

################################################################################


#STEP 1) producing theoretical dL with class or hi-class
print('STEP 1) producing theoretical dL with CLASS')


if MG_bool:
    if MG_params.get('ParametricMG') == True:
        cosmo = cosmology.SetCosmology(cosmo_name, params, cosmo_analysis_type, settings=settings)
        dL_z = lambda z: np.array([( MG_params['Xi0'] + ((1-MG_params['Xi0']) / ((1+zi)**MG_params['n'])) )  * cosmo.luminosity_distance(zi) for zi in z])
    elif  MG_params.get('ParametricMG') == False:
        print('DISCLAIMER: non-minimal modified gravity only works with the hi_class implementation of classy')
        cosmo = cosmology.SetCosmology(cosmo_name, params, cosmo_analysis_type, settings=settings)
        dL_em_z = lambda z: np.array([cosmo.luminosity_distance(zi) for zi in z])
        bkg_cosmo = cosmo.get_background()
        # is it same of bkg_cosmo['lum. dist.'] ?
        Mp2_eff_0 =bkg_cosmo['M*^2_smg'][-1]
        Mp2_eff_z = bkg_cosmo['M*^2_smg']

        # interpolate Mp2_eff_z over redshift

        z_array = bkg_cosmo['z']
        Mp2_eff_z_interp = interp1d(z_array, Mp2_eff_z, kind='cubic', fill_value="extrapolate")

        dL_z =lambda z: np.sqrt(Mp2_eff_0/Mp2_eff_z_interp(z)) * dL_em_z(z)

        #??
else:
    cosmo = cosmology.SetCosmology(cosmo_name, params, cosmo_analysis_type, settings=settings)
    dL_z = lambda z: np.array([cosmo.luminosity_distance(zi) for zi in z])

mock_bool = input_dct['MockBool']          
corner_bool = input_dct['CornerBool']
density_bool = input_dct['DensityBool']
save_posterior_bool = input_dct['SavePosteriorBool']
vp_bool = input_dct['PeculiarCorrectionBool']
plot_vp_bool = input_dct['PlotPeculiarVelocityBool']

# Decide the number of samples to draw
N_sampling = input_dct['NSampling']

# use it to associate a d_L to each redshift for the cosmology you choose
print('STEP 1) Generating mock data for cosmology: ', cosmo_name)
conf1 = mm_gw_pipeline.GenerateCatalog(cosmo_name, dL_z)

#### STEP 2) producing observed dL and uncertanties with gwfish
print('STEP 2) producing observed dL and uncertanties on GW parameters with GWFish')
conf2 = mm_gw_pipeline.GWFishAnalysis(cosmo_name)

#### STEP 3) scatter the {z, d_l} mock data within the Deltad_l errors, and re-centre the error-bars around "observed values"
print('STEP 3) Producing GW posteriors sampling the truncated likelihood')

conf3 = mm_gw_pipeline.GWFishPosterior(cosmo_name, dL_z, N_sampling, produce_mock=mock_bool, cosmo_model=cosmo,
                                       cornerplt=corner_bool, densityplt=density_bool, save_posterior=save_posterior_bool,
                                       include_vp=vp_bool, PlotPeculiar=plot_vp_bool)

#conf5 = mm_gw_pipeline.RandomizeDatset(cosmo_name, plot_
# flag=True, noiseless=noiseless_bool, MultiRndm=True)

# conf6 = mm_gw_pipeline.ProduceSamplerData(cosmo_name, sampler_name= 'Cobaya', noiseless=mock_bool)
# conf7 = mm_gw_pipeline.ProduceSamplerData(cosmo_name, sampler_name= 'Montepython', noiseless=mock_bool)