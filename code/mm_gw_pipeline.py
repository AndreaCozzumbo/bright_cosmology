import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import bright_cosmology  # applies the fishermatrix patch to GWFish on import
import GWFish.modules as gw
from bright_cosmology.priors import *
import os
import shutil
import glob
import matplotlib.pyplot as plt
from corner import corner
import read_input
import matplotlib
import dill as pickle
import warnings
import dill as pickle
from getdist import loadMCSamples
import getdist.plots as gdplt
from scipy.stats import norm
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

input_dct = read_input.read()

catalog_file = input_dct['Catalog']

det = input_dct['Detectors']

fisher_parameters = input_dct['FisherParameters']
waveform_model = input_dct['Waveform']
threshold_SNR = eval(input_dct['SNRThresholds'])

dir_det_config = '_' + f'{det}'
ConfigDet =  input_dct['DetectorPath']

detectors = input_dct['DerectorsCofig']
networks  = input_dct['Network']

injectionGWfish = input_dct['Injection4GWfish'] + f'{dir_det_config}'

output_GWFish = input_dct['CosmoResults'] + f'{dir_det_config}'
results_GWFish = input_dct['GWFishResults']

output_noprior_analysis = input_dct['OutputNoPrior'] + f'{dir_det_config}'

#cosmo_result_dir = input_dct['CosmoResults'] + f'{dir_det_config}' 
mock_dir = input_dct['MockPath'] + f'{dir_det_config}' 

MCMCdir = input_dct['PathMCMC']

corner_lbs = input_dct['CornerLabels']

dL_prior_lowerbound = input_dct['dLLower']
dL_prior_upperbound = input_dct['dLUpper']

theta_jn_lowerbound = input_dct['theta_jnLower']
theta_jn_upperbound = input_dct['theta_jnUpper']

c = 299792.458

rng = np.random.default_rng(seed=5)

def gaus(x,a,x0,sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Directory created: ', path)
    else:
        pass
        #print('Directory already exists: ', path)
    

CORNER_KWARGS = dict(
    bins = 50, # number of bins for histograms
    smooth = 0.99, # smooths out contours. 
    plot_datapoints = False, # choose if you want datapoints
    truth_color='red',
    label_kwargs = dict(fontsize = 16), # font size for labels
    show_titles = True, #choose if you want titles on top of densities.
    title_kwargs = dict(fontsize = 12), # font size for title
    plot_density = True,
    title_quantiles = [0.16, 0.5, 0.84],  # add quantiles to plot densities for 1d hist
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), # 1, 2 and 3 sigma contours for 2d plots
    fill_contours = True, #decide if you want to fill the contours
    max_n_ticks = 3, # set a limit to ticks in the x-y axes.
    title_fmt=".2f"
    )


def GenerateCatalog(cosmology, dL_function):

    cosmo_model = cosmology

    input_file = input_dct['Injection4GWfish']+ '_catalogs' + "/" + catalog_file
    input_data = pd.read_hdf(input_file)

    data = {}
    data['chirp_mass'] = input_data['chirp_mass']
    data['mass_ratio'] = input_data['mass_ratio']
    data['luminosity_distance'] = dL_function(input_data['redshift'])
    data['dec'] = input_data['dec']
    data['ra'] = input_data['ra']
    data['theta_jn'] = input_data['theta_jn']
    data['psi'] = input_data['psi']
    data['geocent_time'] = input_data['geocent_time']
    data['phase'] = input_data['phase']
    data['redshift'] = input_data['redshift']
    data['event'] = input_data['event']

    create_dir(injectionGWfish)

    df = pd.DataFrame(data=data)
    df.to_hdf(injectionGWfish + '/injections_4gwfish_%s.hdf5' % str(cosmo_model), mode='w', key='root')
    
    return 1


def GWFishAnalysis(cosmology, external_file=None):

     # [min. individual SNR to be included in PE, min. network SNR for detection]

    if external_file is None:

        pop_file = injectionGWfish +'/injections_4gwfish_%s.hdf5' % str(cosmology)
    
    else:
        pop_file = external_file

    population = 'BNS'

    networks_ids = json.loads(networks)

    parameters = pd.read_hdf(pop_file)
    print(parameters)

    catalog_config = catalog_file.split('/')[-1].split('_')[1].split('.')[0]

    
    kwargs_network = {'detector_ids': detectors}

    if threshold_SNR is not None:
        kwargs_network['detection_SNR'] = threshold_SNR
    if ConfigDet!= ' ':
        kwargs_network['config'] = ConfigDet

    network = gw.detection.Network(**kwargs_network)
    
    path_out = results_GWFish + f'/{cosmology}'
    
    # - 1 is the redshift
    #my_format = " ".join(["%.3E" for _ in range(parameters.shape[1] - 1)]) + " %s " + " ".join(["%.3E" for _ in range(np.shape(fisher_parameters)[0])])

    gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                        parameter_values  = parameters,
                                        fisher_parameters = fisher_parameters, 
                                        sub_network_ids_list = networks_ids,
                                        population_name = population,
                                        waveform_model = waveform_model,
                                        save_path = path_out,
                                        save_matrices = True,
                                        use_duty_cycle = True)
    
    # CREATE SAME FILES BUT NEW DIRECTORY 

    move_results = os.getcwd() + '/' + output_GWFish + f'/{cosmology}'

    create_dir(move_results)

    file_to_move = glob.glob(path_out+'/*')

    for i in range(len(file_to_move)):
        appendix = file_to_move[i].split('.')[-1]

        file_cropped = file_to_move[i].split('/')[-1].split('_')
        cut = np.where(np.array(file_cropped)==detectors[0].split('_')[0])[0][0]
        file_name_change = '_'.join(file_cropped[0:cut])
        shutil.copy(file_to_move[i], move_results+f'/{file_name_change}_{catalog_config}_{cosmology}.'+f'{appendix}')

    # Remove everything inside 'gwfish_temp_results' to prevent copies

    parent_dir = os.path.dirname(path_out)  # Parent folder of path_out
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and item != results_GWFish:  # Exclude 'results' directory
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)

    return 1


def GWFishPosterior(cosmology, dL_z, sampling, produce_mock=True, cosmo_model=None, priorplt=False,
                   cornerplt=False, postplt= False, densityplt=False, save_posterior=False, include_vp=False, PlotPeculiar=False):

    catalog_config = catalog_file.split('/')[-1].split('_')[1].split('.')[0]

    if produce_mock:

        file_path = f'Errors_{catalog_config}_{cosmology}.txt'
        with open(output_GWFish+f'/{cosmology}/'+file_path, 'r') as file:
            header_line = file.readline().strip()  
            if header_line.startswith('#') or header_line.startswith('//'): 
                header_line = header_line[1:].strip()
            lbs = header_line.replace(',', ' ').split()

        means = pd.read_csv(output_GWFish+f'/{cosmology}/'+file_path, names = lbs, skiprows = 1, delimiter = ' ')
        cov_matrix = np.load(output_GWFish+f'/{cosmology}/'+f'/inv_fisher_matrices_{catalog_config}_{cosmology}.npy')

        lbs_mns = fisher_parameters

        mns = means[lbs_mns]
        ns = len(mns)
        N = sampling
        zz = np.linspace(0.0001, 5, 1000)
        zmax = max(means['redshift'])
        
        m1min = input_dct['ComponentMassLower']
        m2min = input_dct['ComponentMassLower']

        m1max = input_dct['ComponentMassUpper'] * (1+zmax)
        m2max = input_dct['ComponentMassUpper'] * (1+zmax)

        plotPrior = priorplt
        CornerPlot = cornerplt
        plotPosteriros = postplt
        DensityPlot = densityplt

        event_list = []
        z_list  = []
        density_list = []
        skipped_events = []
        posterior_list = []
        dl_inject = []
        Ddl_inject = []

        min_array = eval(input_dct['MinTruncated'])
        max_array = eval(input_dct['MaxTruncated'])

        for event in tqdm(np.arange(ns)):

            mns_ev = mns.iloc[event].to_numpy()
            cov_ev = cov_matrix[event, :, :]
            rng = np.random.default_rng(seed=43)

            fisher_samples = pd.DataFrame(rng.multivariate_normal(mns_ev, cov_ev, N), columns = fisher_parameters)
            
            # TRUNCATED SAMPLING

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # Convert warnings into exceptions
                    
                    samples_truncated_mvn = get_truncated_likelihood_samples(
                        fisher_parameters, mns_ev, cov_ev, N, 
                        min_array=min_array, max_array=max_array
                    )

            except Warning as w:  # Catch warnings
                print(f"Warning in truncated evaluation: {w}")
                skipped_events.append([means['event'][event], means['redshift'][int(event)]])
                continue  # Skip to the next iteration

            except Exception as e:  # Catch other exceptions
                print(f"Error in truncated evaluation: {e}")
                skipped_events.append([means['event'][event], means['redshift'][int(event)]])
                continue


            priors_dict = get_default_priors_dict(fisher_parameters) 

            # Chirp mass prior

            if input_dct['ChirpMassPriorType'] == priors_dict['chirp_mass']['prior_type'] or input_dct['ChirpMassPriorType'] in [None, 'None']:

                priors_dict['chirp_mass']['prior_type'] = priors_dict['chirp_mass']['prior_type']

                if input_dct['ComponentMassLower'] not in [None, 'None']:

                    priors_dict['chirp_mass']['lower_prior_bound'] = (m1min * m2min)**(3/5) / (m1min + m2min)**(1/5)

                if input_dct['ComponentMassUpper'] not in [None, 'None']:

                    priors_dict['chirp_mass']['upper_prior_bound'] = (m1max * m2max)**(3/5) / (m1max + m2max)**(1/5)


            else:   #should add the try and except here for the type of prior
                priors_dict['chirp_mass']['prior_type'] = input_dct['ChirpMassPriorType']
            
                if input_dct['ComponentMassLower'] not in [None, 'None']:

                    priors_dict['chirp_mass']['lower_prior_bound'] = (m1min * m2min)**(3/5) / (m1min + m2min)**(1/5)

                if input_dct['ComponentMassUpper'] not in [None, 'None']:

                    priors_dict['chirp_mass']['upper_prior_bound'] = (m1max * m2max)**(3/5) / (m1max + m2max)**(1/5)

            # Luminosity distance prior

            if input_dct['dLPriorType'] == priors_dict['luminosity_distance']['prior_type'] or input_dct['dLPriorType'] in [None, 'None']:
                
                if dL_prior_lowerbound not in [None, 'None']:

                    priors_dict['luminosity_distance']['lower_prior_bound'] = dL_prior_lowerbound

                if dL_prior_upperbound not in [None, 'None']:

                    priors_dict['luminosity_distance']['upper_prior_bound'] = dL_prior_upperbound

            elif input_dct['dLPriorType'] == 'uniform_in_differential_comoving_volume_Class':

                if dL_prior_lowerbound not in [None, 'None']:

                    priors_dict['luminosity_distance']['lower_prior_bound'] = dL_prior_lowerbound

                if dL_prior_upperbound not in [None, 'None']:

                    priors_dict['luminosity_distance']['upper_prior_bound'] = dL_prior_upperbound

                dL_interp = np.interp(samples_truncated_mvn['luminosity_distance'].to_numpy(), dL_z(zz), zz)

                z_lowerbound = np.interp(priors_dict['luminosity_distance']['lower_prior_bound'], dL_z(zz), zz)
                z_upperbound = np.interp(priors_dict['luminosity_distance']['upper_prior_bound'], dL_z(zz), zz)

                priors_dict['luminosity_distance'] = {
                    'prior_type': f"{input_dct['dLPriorType']}",
                    'lower_prior_bound': z_lowerbound,
                    'upper_prior_bound': z_upperbound,
                }

            else:
            
                priors_dict['luminosity_distance']['prior_type'] = input_dct['dLPriorType']

                if dL_prior_lowerbound not in [None, 'None']:

                    priors_dict['luminosity_distance']['lower_prior_bound'] = dL_prior_lowerbound

                if dL_prior_upperbound not in [None, 'None']:

                    priors_dict['luminosity_distance']['upper_prior_bound'] = dL_prior_upperbound
            
            # theta_jn prior
            if input_dct['ThetaPriorType'] == priors_dict['theta_jn']['prior_type'] or input_dct['ThetaPriorType'] in [None, 'None']:

                priors_dict['theta_jn']['prior_type'] = priors_dict['theta_jn']['prior_type']
                
                if theta_jn_lowerbound not in [None, 'None']:

                    if 0 < means['theta_jn'][event] < np.pi/2 :
                        priors_dict['theta_jn']['lower_prior_bound'] = theta_jn_lowerbound*np.pi/180.
                                    
                    else:
                        priors_dict['theta_jn']['lower_prior_bound'] = (180-theta_jn_upperbound)*np.pi/180

                if theta_jn_upperbound not in [None, 'None']:

                    if 0 < means['theta_jn'][event] < np.pi/2 :

                        priors_dict['theta_jn']['upper_prior_bound'] = theta_jn_upperbound*np.pi/180.
                                    
                    else:
                        priors_dict['theta_jn']['upper_prior_bound'] = (180-theta_jn_lowerbound)*np.pi/180

            else:

                priors_dict['theta_jn']['prior_type'] = input_dct['ThetaPriorType']

                if theta_jn_lowerbound not in [None, 'None']:

                    if 0 < means['theta_jn'][event] < np.pi/2 :
                        priors_dict['theta_jn']['lower_prior_bound'] = theta_jn_lowerbound*np.pi/180.
                                    
                    else:
                        priors_dict['theta_jn']['lower_prior_bound'] = (180-theta_jn_upperbound)*np.pi/180

                if theta_jn_upperbound is not None:

                    if 0 < means['theta_jn'][event] < np.pi/2 :

                        priors_dict['theta_jn']['upper_prior_bound'] = theta_jn_upperbound*np.pi/180.
                                    
                    else:
                        priors_dict['theta_jn']['upper_prior_bound'] = (180-theta_jn_lowerbound)*np.pi/180


            # PRIOR EVALUATION and POSTERIOR SAMPLING

            if input_dct['dLPriorType'] == 'uniform_in_differential_comoving_volume_Class':

                try:
                    samples_from_posterior, prior_samples = get_posteriors_samples(fisher_parameters, samples_truncated_mvn, N, priors_dict, cosmology=cosmo_model, dL_z=dL_interp)
                except Exception as e:
                    print('Error in the prior evaluation:', e)
                    skipped_events.append([means['event'][event], means['redshift'][int(event)]])
                    continue
            else:
                
                try:
                    samples_from_posterior, prior_samples = get_posteriors_samples(fisher_parameters, samples_truncated_mvn, N)
                
                except Exception as e:
                    print('Error in the prior evaluation:', e)
                    skipped_events.append([means['event'][event], means['redshift'][int(event)]])
                    continue

            evprint = means['event'][event]
            z_save = means['redshift'][int(event)]

            event_save = means['event'][event]

            if plotPrior:

                #plot histogram prior 

                plt.hist(prior_samples, bins = 50)
                plt.xlabel('prior')
                plt.yscale('log')
                plt.title(f'Event {evprint} at z={z_save}')
                plot_dir = mock_dir + f'/{catalog_config}/{cosmology}/plots/prior'
                create_dir(plot_dir)
                plt.savefig(plot_dir + f'/priorplot_'+ evprint + '.pdf',dpi=600, bbox_inches='tight', facecolor='white')
                #plt.show()
                plt.close()

                            
            if CornerPlot:

                sample_labels = ['likelihood', 'truncated lkl', 'posteriror']
                    
                fig = corner(fisher_samples, labels = corner_lbs, truths = mns_ev, color = 'black', alpha = 0.5, **CORNER_KWARGS)
                old_fig = corner(samples_truncated_mvn, labels = corner_lbs, truths = mns_ev, color = 'orange', alpha = 0.5, **CORNER_KWARGS, fig = fig)
                new_fig = corner(samples_from_posterior, labels = corner_lbs, truths = mns_ev, color = 'red', alpha = 0.5, **CORNER_KWARGS, fig = fig)
               
                plt.suptitle(f'Event: {evprint}')
                
                plot_dir = mock_dir + f'/{catalog_config}/{cosmology}/plots/corner'
                create_dir(plot_dir)
                plt.savefig(plot_dir + f'/cornerplot_'+ evprint + '.pdf',dpi=600, bbox_inches='tight', facecolor='white')
                #plt.show()
                plt.close()

            if plotPosteriros:

                evprint = means['event'][event]

                # Set font size for ticks
                matplotlib.rc('xtick', labelsize=8)
                matplotlib.rc('ytick', labelsize=8)

                # Create subplots
                fig, axs = plt.subplots(2, 3, figsize=(10, 10))
                fig.suptitle(f'Posterior distributions Event: {evprint}')

                # Histogram plots
                params = fisher_parameters
                positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

                for param, pos in zip(params, positions):
                    counts, edges, _ = axs[pos].hist(samples_from_posterior[param].to_numpy(), bins=50)
                    axs[pos].set_xlabel(param, fontsize=8)
                    
                    # Plot injected values
                    axs[pos].axvline(x=mns_ev[params.index(param)], color='red', label='injected')
                    
                    # Plot new mean values
                    axs[pos].axvline(x=samples_from_posterior[param].mean(), color='orange', label='new mean')
                    
                    # Plot max posterior values
                    mode_index = np.argmax(counts)
                    axs[pos].axvline(x=0.5 * (edges[mode_index] + edges[mode_index + 1]), color='lightgreen', label='max posterior')

                # Adjust layout and add legend
                plt.legend(bbox_to_anchor=(1., 0.5), loc='center left', fontsize=8)
                plt.tight_layout()

                # Save plot
                target_dir = mock_dir + f'/{catalog_config}/{cosmology}/plots/posteriors'
                create_dir(target_dir)
                plt.savefig(f'{target_dir}/GW_posteriors_{evprint}.pdf', dpi=600, bbox_inches='tight', facecolor='white')

                # Close figure
                plt.close()
   
            
            
            # SAVE ALL POSTERIORS
            
            weights, edges, _ = plt.hist(samples_from_posterior['luminosity_distance'].to_numpy(), bins=100, density=True)
            plt.close()
            
            out_path_posteriors = mock_dir + f'/{cosmology}'
            create_dir(out_path_posteriors)
            final_samples = np.zeros((3,len(samples_from_posterior['luminosity_distance'].to_numpy())))
            final_samples[0, :] = 1
            final_samples[2,:] = samples_from_posterior['luminosity_distance'].to_numpy()
            # CREATE PARAMFILE
            param_list = ['d_L']
            paramfile = [param_list[ii] + '\t' + param_list[ii] for ii in range(len(param_list))]
            np.savetxt(out_path_posteriors + f'/posteriors_tmp.paramnames', paramfile, fmt='%s')
            
            np.savetxt(out_path_posteriors+ f'/posteriors_tmp.txt', np.transpose(final_samples), fmt='%i %.6e %.6e')
            settings = {'smooth_scale_1D' : input_dct['KDESmoothing']}
            sample_getdist=gdplt.loadMCSamples(out_path_posteriors + f'/posteriors_tmp', settings=settings)
            density = gdplt.MCSamples.get1DDensityGridData(sample_getdist, 0)
            event_list.append(event_save)
            z_list.append(z_save)
            
            density_list.append(density)
            if DensityPlot:
                plt.rc('font', size=25, family='serif')
                plt.rc('text', usetex=True)
                fig = plt.figure(figsize=(10, 10))
                xx = np.linspace(min(samples_from_posterior['luminosity_distance'].to_numpy()), max(samples_from_posterior['luminosity_distance'].to_numpy()),300)
                plt.hist(samples_from_posterior['luminosity_distance'].to_numpy(), bins=100, density=True, alpha=0.7)
                plt.plot(xx, max(weights)*density(xx), lw=3, label= 'Interpolating density')
                #plt.plot(xx, max(weights)*density2(xx), label= 'Interpolating density bw=0.3')
                plt.legend()
                plt.title(f'{evprint} @ z={z_save}')
                
                plot_dir = mock_dir + f'/{catalog_config}/{cosmology}/plots/density'
                create_dir(plot_dir)
                plt.tight_layout()
                plt.xlabel(r'$z$')
                plt.ylabel(r'$p(d_L)$')
                plt.savefig(plot_dir + f'/densityplot_'+ evprint + '.pdf',dpi=600, bbox_inches='tight', facecolor='white')
                #plt.show()
                plt.close()
            
            # just to save the posteriors
            posterior_list.append(samples_from_posterior['luminosity_distance'].to_numpy())
            dl_inject.append(means['luminosity_distance'].to_numpy()[event])
            Ddl_inject.append(means['err_luminosity_distance'].to_numpy()[event])


            
        out_path_kde = (MCMCdir+f'/{cosmology}')
        create_dir(out_path_kde)

        if '.' in catalog_config:
            
            catalog_config=catalog_config.split('.')[0]
            
        if include_vp:
            print('Peculiar velocity correction on going')
            print('Starting the peculiar velocity analysis\n')
            # include vp correction
            z_thr = input_dct['z_vp_threshold']
            
            dt = {}
            dt['name'] = event_list
            dt['z'] = z_list
            dt['posterior'] = posterior_list
            dt['kde'] = density_list
            dt['dl'] = dl_inject
            dt['Ddl'] = Ddl_inject
            dataframe_data = pd.DataFrame(data=dt)                
            sorted_df = dataframe_data.sort_values(by=['z'])
            idx_cycle = np.where(np.array(sorted_df['z'])<=z_thr)[0]
            list_vp = []
            list_svp = []
            
            if len(idx_cycle) != 0:
                
                for idx, ii in enumerate(idx_cycle):
                    
                    gw_samples = sorted_df['posterior'].to_numpy()[ii]
                    gw_density = sorted_df['kde'].to_numpy()[ii]
                    z_event = sorted_df['z'].to_numpy()[ii]
                    ev = sorted_df['name'].to_numpy()[ii]
                    dl= sorted_df['dl'].to_numpy()[ii]
                    Ddl= sorted_df['Ddl'].to_numpy()[ii]
                    rng = np.random.RandomState(ii)
                    vp_rms = rng.normal(loc=input_dct['Mean_vp'], scale=50) #input_dct['Mean_vp']     #
                    sigma_vp = rng.normal(loc=input_dct['Sigma_vp'], scale=50) #input_dct['Sigma_vp']  #
                    
                    list_vp.append(eval("{:.2f}".format(vp_rms)))
                    list_svp.append(eval("{:.2f}".format(sigma_vp)))
                    sigma_dL_vp = np.median(gw_samples) * sigma_vp / (vp_rms + c * z_event) 
                    #first estimate of the peculiar velocity distribution impedence when combined by 
                    #simple quadrature sum because we need dL_values to be big enough
                    peculiar_Ddl = np.sqrt((Ddl)**2 + sigma_dL_vp**2)
                    if dl-3*peculiar_Ddl <=0:
                        min_bound = 0
                    else:
                        min_bound = dl-3*peculiar_Ddl
                    dL_values = np.linspace(min_bound, dl+3*peculiar_Ddl, 10000)
                    # Gaussian error on dL from the peculiar velocity. It has zero mean: the kernel is
                    # centred on the index that np.convolve(mode='same') treats as zero shift, so that
                    # it widens the density without moving it
                    zero_shift = dL_values[(len(dL_values) - 1) // 2]
                    v_p_distribution = norm(loc=zero_shift, scale=sigma_dL_vp)
                    # Calculate the PDF of the peculiar velocity part
                    pdf_vp = v_p_distribution.pdf(dL_values)
                    # Calculate the PDF of the combined distribution by convolving the PDFs
                    combined_pdf = np.convolve(pdf_vp, gw_density(dL_values), mode='same')
                    # Normalize the combined PDF
                    combined_pdf /= np.sum(combined_pdf)
                    histogram, bin_edges = np.histogram(dL_values, bins=10000, weights=combined_pdf)
                    combined_samples = np.random.choice((bin_edges[:-1] + bin_edges[1:]) / 2, size=1_000_000, p= np.abs(histogram) / np.sum(histogram))
                    samp = np.zeros((3,len(combined_samples)))
                    samp[0, :] = 1
                    samp[2,:] = combined_samples
                    np.savetxt(out_path_posteriors+ f'/posteriors_tmp.txt', np.transpose(samp), fmt='%i %.6e %.6e')
                    
                    settings = {'smooth_scale_1D' : input_dct['KDESmoothing']}
                    getdist=gdplt.loadMCSamples(out_path_posteriors+ f'/posteriors_tmp', settings=settings)
                    density_final = gdplt.MCSamples.get1DDensityGridData(getdist, 0)
                    
                    dataframe_data['kde'][np.where(dataframe_data['name']==ev)[0][0]] = density_final
                    
                    #Plot the combined distribution
                    if PlotPeculiar:
                        plt.figure(figsize=(10, 10))

                        plt.plot(dL_values, density_final(dL_values), label=r'Combined kde with Gaussian $v_{\mathrm{peculiar}}$', color='green')
                        plt.plot(dL_values, gw_density(dL_values), label='Old kde', color='red')
                        plt.xlabel(r'$d_L$')
                        plt.ylabel('Density')
                        plt.title(f'Event : {ev} at $z$={z_event} | $z_t$={z_thr} $v_p$={vp_rms:.2f} $\sigma_v$={sigma_vp:.2f}')
                        plt.legend(loc='upper right')
                        plt.xlim(np.min(gw_samples), np.max(gw_samples))
                    
                        plot_dir = mock_dir + f'/{catalog_config}/{cosmology}/plots/peculiar_velocity'
                        create_dir(plot_dir)
                        plt.savefig(plot_dir + f'/peculiar_velocity_'+ ev + '.pdf',dpi=600, bbox_inches='tight', facecolor='white')
                        #plt.show()
                        plt.close()
                
                density_vp = list(dataframe_data['kde'])
                if save_posterior:
                    save_density = [event_list, z_list, posterior_list, density_list, dl_inject, Ddl_inject, density_vp]
                else:
                    save_density = [event_list, z_list, density_list, dl_inject, Ddl_inject, density_vp]

                sorted_name_list = sorted_df['name'].to_numpy()[idx_cycle[0]:idx_cycle[-1]+1]
            else:
                if save_posterior:
                    save_density = [event_list, z_list, posterior_list, density_list, dl_inject, Ddl_inject]
                else:
                    save_density = [event_list, z_list, density_list, dl_inject, Ddl_inject]
            
                sorted_name_list = None
        else:
            # the kde are the same as before
            if save_posterior:
                save_density = [event_list, z_list, posterior_list, density_list, dl_inject, Ddl_inject]
            else:
                save_density = [event_list, z_list, density_list, dl_inject, Ddl_inject]
            
            sorted_name_list = None
        
        if cosmology == 'LCDM':
            cosmology_save = 'LCDM_Planck18'
        else:
            cosmology_save = cosmology
        
        create_dir(out_path_kde + f'/{det}_{cosmology_save}_{catalog_config}')
        with open(out_path_kde + f'/{det}_{cosmology_save}_{catalog_config}/{det}_{cosmology_save}_{catalog_config}.pkl', 'wb') as handle:
            
            pickle.dump(save_density, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_file=read_input.write_in_file(out_path_kde + f'/{det}_{cosmology_save}_{catalog_config}/log_file.txt')    
        os.remove(mock_dir + f'/{cosmology}/posteriors_tmp.txt')
        
        
        print(f'Max z seen: {max(z_list)}\n N. of events: {len(event_list)}')
        print(f'Events corrected: {sorted_name_list}')
        print(f'v_peculiar: {list_vp}')
        print(f'sigma v_peculiar: {list_svp}')
        print('Skipped events [name, redshift]:', skipped_events)
        with open(out_path_kde + f'/{det}_{cosmology_save}_{catalog_config}/log_file.txt', 'a') as file:
            file.writelines('\n')
            file.writelines(f'Max z seen: {max(z_list)}\nN. of events: {len(event_list)}')
            file.writelines('\n')
            file.writelines(f'Events corrected: {sorted_name_list}')
            file.writelines('\n')
            file.writelines(f'v_peculiar: {list_vp}')
            file.writelines('\n')
            file.writelines(f'sigma v_peculiar: {list_svp}')
            file.writelines('\n')
            file.writelines(f'Skipped events [name, redshift]: {skipped_events}')
        del(event_list, z_list, density_list, cosmology, cosmo_model, log_file)

    else:
        
        file_path = f'Errors_{catalog_config}_{cosmology}.txt'
        
        with open(file_path, 'r') as file:
            header_line = file.readline().strip()  
            if header_line.startswith('#') or header_line.startswith('//'): 
                header_line = header_line[1:].strip()
            lbs = header_line.replace(',', ' ').split()

        data_in = pd.read_csv(file_path, names = lbs, delimiter = ' ', low_memory = False, skiprows = 1)
        cov = np.load(output_GWFish+'/Inv_Fishers_%s.npy' %str(cosmology))
        
        data_to_save = {}
        data_to_save['chirp_mass'] = data_in['chirp_mass']
        data_to_save['mass_ratio'] = data_in['mass_ratio']
        data_to_save['luminosity_distance'] = data_in['luminosity_distance']
        data_to_save['dec'] = data_in['dec']
        data_to_save['ra'] = data_in['ra']
        data_to_save['theta_jn'] = data_in['theta_jn']
        data_to_save['psi'] = data_in['psi']
        data_to_save['geocent_time'] = data_in['geocent_time']
        data_to_save['phase'] = data_in['phase']
        data_to_save['redshift'] = data_in['redshift']

        path_output_prior = os.getcwd() + '/'+ output_noprior_analysis

        create_dir(path_output_prior)

        df = pd.DataFrame(data=data_to_save)
        df.to_hdf(path_output_prior + '/gwfish_analysis_%s.hdf5' % str(cosmology), mode='w', key='root')

        np.save(path_output_prior+'/covariance_%s.npy' % str(cosmology), cov)
        
    return 1
