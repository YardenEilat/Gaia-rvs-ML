import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def read_rvs_csv(file_path, type_dict=None):
    df = pd.read_csv(file_path, skiprows=1541, dtype=type_dict)
    return df

def save_ids_to_csv(df, file_path):
    df.to_csv(file_path, columns=['source_id'], index=False)
    

def check_source_id(i, source_ids, temp_ids):
    if temp_ids[i] not in source_ids:
        return i
    


if __name__=="__main__":
    type_dict = {'solution_id': 'int64', 'source_id': 'int64', 'classprob_dsc_combmod_quasar': 'float64', 'classprob_dsc_combmod_galaxy': 'float64', 'classprob_dsc_combmod_star': 'float64', 'classprob_dsc_combmod_whitedwarf': 'float64', 'classprob_dsc_combmod_binarystar': 'float64', 'classprob_dsc_specmod_quasar': 'float64', 'classprob_dsc_specmod_galaxy': 'float64', 'classprob_dsc_specmod_star': 'float64', 'classprob_dsc_specmod_whitedwarf': 'float64', 'classprob_dsc_specmod_binarystar': 'float64', 'classprob_dsc_allosmod_quasar': 'float64', 'classprob_dsc_allosmod_galaxy': 'float64', 'classprob_dsc_allosmod_star': 'float64', 'teff_gspphot': 'float64', 'teff_gspphot_lower': 'float64', 'teff_gspphot_upper': 'float64', 'logg_gspphot': 'float64', 'logg_gspphot_lower': 'float64', 'logg_gspphot_upper': 'float64', 'mh_gspphot': 'float64', 'mh_gspphot_lower': 'float64', 'mh_gspphot_upper': 'float64', 'distance_gspphot': 'float64', 'distance_gspphot_lower': 'float64', 'distance_gspphot_upper': 'float64', 'azero_gspphot': 'float64', 'azero_gspphot_lower': 'float64', 'azero_gspphot_upper': 'float64', 'ag_gspphot': 'float64', 'ag_gspphot_lower': 'float64', 'ag_gspphot_upper': 'float64', 'abp_gspphot': 'float64', 'abp_gspphot_lower': 'float64', 'abp_gspphot_upper': 'float64', 'arp_gspphot': 'float64', 'arp_gspphot_lower': 'float64', 'arp_gspphot_upper': 'float64', 'ebpminrp_gspphot': 'float64', 'ebpminrp_gspphot_lower': 'float64', 'ebpminrp_gspphot_upper': 'float64', 'mg_gspphot': 'float64', 'mg_gspphot_lower': 'float64', 'mg_gspphot_upper': 'float64', 'radius_gspphot': 'float64', 'radius_gspphot_lower': 'float64', 'radius_gspphot_upper': 'float64', 'logposterior_gspphot': 'float64', 'mcmcaccept_gspphot': 'float64', 'libname_gspphot': 'str', 'teff_gspspec': 'float64', 'teff_gspspec_lower': 'float64', 'teff_gspspec_upper': 'float64', 'logg_gspspec': 'float64', 'logg_gspspec_lower': 'float64', 'logg_gspspec_upper': 'float64', 'mh_gspspec': 'float64', 'mh_gspspec_lower': 'float64', 'mh_gspspec_upper': 'float64', 'alphafe_gspspec': 'float64', 'alphafe_gspspec_lower': 'float64', 'alphafe_gspspec_upper': 'float64', 'fem_gspspec': 'float64', 'fem_gspspec_lower': 'float64', 'fem_gspspec_upper': 'float64', 'fem_gspspec_nlines': 'float64', 'fem_gspspec_linescatter': 'float64', 'sife_gspspec': 'float64', 'sife_gspspec_lower': 'float64', 'sife_gspspec_upper': 'float64', 'sife_gspspec_nlines': 'float64', 'sife_gspspec_linescatter': 'float64', 'cafe_gspspec': 'float64', 'cafe_gspspec_lower': 'float64', 'cafe_gspspec_upper': 'float64', 'cafe_gspspec_nlines': 'float64', 'cafe_gspspec_linescatter': 'float64', 'tife_gspspec': 'float64', 'tife_gspspec_lower': 'float64', 'tife_gspspec_upper': 'float64', 'tife_gspspec_nlines': 'float64', 'tife_gspspec_linescatter': 'float64', 'mgfe_gspspec': 'float64', 'mgfe_gspspec_lower': 'float64', 'mgfe_gspspec_upper': 'float64', 'mgfe_gspspec_nlines': 'float64', 'mgfe_gspspec_linescatter': 'float64', 'ndfe_gspspec': 'float64', 'ndfe_gspspec_lower': 'float64', 'ndfe_gspspec_upper': 'float64', 'ndfe_gspspec_nlines': 'float64', 'ndfe_gspspec_linescatter': 'float64', 'feiim_gspspec': 'float64', 'feiim_gspspec_lower': 'float64', 'feiim_gspspec_upper': 'float64', 'feiim_gspspec_nlines': 'float64', 'feiim_gspspec_linescatter': 'float64', 'sfe_gspspec': 'float64', 'sfe_gspspec_lower': 'float64', 'sfe_gspspec_upper': 'float64', 'sfe_gspspec_nlines': 'float64', 'sfe_gspspec_linescatter': 'float64', 'zrfe_gspspec': 'float64', 'zrfe_gspspec_lower': 'float64', 'zrfe_gspspec_upper': 'float64', 'zrfe_gspspec_nlines': 'float64', 'zrfe_gspspec_linescatter': 'float64', 'nfe_gspspec': 'float64', 'nfe_gspspec_lower': 'float64', 'nfe_gspspec_upper': 'float64', 'nfe_gspspec_nlines': 'float64', 'nfe_gspspec_linescatter': 'float64', 'crfe_gspspec': 'float64', 'crfe_gspspec_lower': 'float64', 'crfe_gspspec_upper': 'float64', 'crfe_gspspec_nlines': 'float64', 'crfe_gspspec_linescatter': 'float64', 'cefe_gspspec': 'float64', 'cefe_gspspec_lower': 'float64', 'cefe_gspspec_upper': 'float64', 'cefe_gspspec_nlines': 'float64', 'cefe_gspspec_linescatter': 'float64', 'nife_gspspec': 'float64', 'nife_gspspec_lower': 'float64', 'nife_gspspec_upper': 'float64', 'nife_gspspec_nlines': 'float64', 'nife_gspspec_linescatter': 'float64', 'cn0ew_gspspec': 'float64', 'cn0ew_gspspec_uncertainty': 'float64', 'cn0_gspspec_centralline': 'float64', 'cn0_gspspec_width': 'float64', 'dib_gspspec_lambda': 'float64', 'dib_gspspec_lambda_uncertainty': 'float64', 'dibew_gspspec': 'float64', 'dibew_gspspec_uncertainty': 'float64', 'dibewnoise_gspspec_uncertainty': 'float64', 'dibp0_gspspec': 'float64', 'dibp2_gspspec': 'float64', 'dibp2_gspspec_uncertainty': 'float64', 'dibqf_gspspec': 'float64', 'flags_gspspec': 'str', 'logchisq_gspspec': 'float64', 'ew_espels_halpha': 'float64', 'ew_espels_halpha_uncertainty': 'float64', 'ew_espels_halpha_flag': 'float64', 'ew_espels_halpha_model': 'float64', 'classlabel_espels': 'str', 'classlabel_espels_flag': 'float64', 'classprob_espels_wcstar': 'float64', 'classprob_espels_wnstar': 'float64', 'classprob_espels_bestar': 'float64', 'classprob_espels_ttauristar': 'float64', 'classprob_espels_herbigstar': 'float64', 'classprob_espels_dmestar': 'float64', 'classprob_espels_pne': 'float64', 'azero_esphs': 'float64', 'azero_esphs_uncertainty': 'float64', 'ag_esphs': 'float64', 'ag_esphs_uncertainty': 'float64', 'ebpminrp_esphs': 'float64', 'ebpminrp_esphs_uncertainty': 'float64', 'teff_esphs': 'float64', 'teff_esphs_uncertainty': 'float64', 'logg_esphs': 'float64', 'logg_esphs_uncertainty': 'float64', 'vsini_esphs': 'float64', 'vsini_esphs_uncertainty': 'float64', 'flags_esphs': 'float64', 'spectraltype_esphs': 'str', 'activityindex_espcs': 'float64', 'activityindex_espcs_uncertainty': 'float64', 'activityindex_espcs_input': 'str', 'teff_espucd': 'float64', 'teff_espucd_uncertainty': 'float64', 'flags_espucd': 'float64', 'radius_flame': 'float64', 'radius_flame_lower': 'float64', 'radius_flame_upper': 'float64', 'lum_flame': 'float64', 'lum_flame_lower': 'float64', 'lum_flame_upper': 'float64', 'mass_flame': 'float64', 'mass_flame_lower': 'float64', 'mass_flame_upper': 'float64', 'age_flame': 'float64', 'age_flame_lower': 'float64', 'age_flame_upper': 'float64', 'flags_flame': 'float64', 'evolstage_flame': 'float64', 'gravredshift_flame': 'float64', 'gravredshift_flame_lower': 'float64', 'gravredshift_flame_upper': 'float64', 'bc_flame': 'float64', 'mh_msc': 'float64', 'mh_msc_upper': 'float64', 'mh_msc_lower': 'float64', 'azero_msc': 'float64', 'azero_msc_upper': 'float64', 'azero_msc_lower': 'float64', 'distance_msc': 'float64', 'distance_msc_upper': 'float64', 'distance_msc_lower': 'float64', 'teff_msc1': 'float64', 'teff_msc1_upper': 'float64', 'teff_msc1_lower': 'float64', 'teff_msc2': 'float64', 'teff_msc2_upper': 'float64', 'teff_msc2_lower': 'float64', 'logg_msc1': 'float64', 'logg_msc1_upper': 'float64', 'logg_msc1_lower': 'float64', 'logg_msc2': 'float64', 'logg_msc2_upper': 'float64', 'logg_msc2_lower': 'float64', 'ag_msc': 'float64', 'ag_msc_upper': 'float64', 'ag_msc_lower': 'float64', 'logposterior_msc': 'float64', 'mcmcaccept_msc': 'float64', 'mcmcdrift_msc': 'float64', 'flags_msc': 'float64', 'neuron_oa_id': 'float64', 'neuron_oa_dist': 'float64', 'neuron_oa_dist_percentile_rank': 'float64', 'flags_oa': 'float64'}
    
    with np.load('all_weirdness_scores_highsnr_no_nn_no_duplicates.npz', allow_pickle=True) as d:
        source_ids = d['source_ids']
    source_ids = set(source_ids)
    spectra_folder_path = 'Gaia/gdr3/Astrophysical_parameters/astrophysical_parameters'
    full_df = pd.DataFrame()
    for file in tqdm(os.listdir(spectra_folder_path)):
        print(f'{file}')
        if file.endswith('gz'):
            print('unzipping...')
            os.system('gunzip ' + os.path.join(spectra_folder_path, file))
            file = file[:-3]
            print(file)
        if file.endswith('csv'):
            df_temp = read_rvs_csv(os.path.join(spectra_folder_path, file), type_dict)
            # filter rows with no gspspec solution
            df_temp = df_temp[~df_temp['flags_gspspec'].isna()]
            df_temp = df_temp.reset_index(drop=True)
            source_ids_temp = df_temp['source_id'].values
            # delete rows with irrelevant source ids
            inds_to_drop = []
            # for i in tqdm(range(len(source_ids_temp))):
            #     if source_ids_temp[i] not in source_ids:
            #         inds_to_drop.append(i)
            # replace the for loop by multiprocessing
            with mp.Pool(4) as p:
                inds_to_drop = p.map(partial(check_source_id, source_ids=source_ids, temp_ids=source_ids_temp), tqdm(range(len(source_ids_temp))))
            inds_to_drop = list(filter(lambda x: x is not None, inds_to_drop))
            df_temp = df_temp.drop(np.array(inds_to_drop))
            full_df = full_df.append(df_temp, ignore_index=True)
            print(f'The length of the full df is {len(full_df)}')
            full_df.to_pickle('astroparams_df.pkl')
            os.system(f'rm {os.path.join(spectra_folder_path, file)}')
    full_df.to_pickle('astroparams_df.pkl')
    save_ids_to_csv(full_df, 'rvs_source_ids.csv')

    print('DONE')