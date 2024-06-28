import numpy as np

d = np.load('filtered_nans_fluxes_and_ids_highsnr.npz',
            allow_pickle=True, mmap_mode='r+')
flux_source_ids = d['source_ids']
# find duplicates
duplicates_to_remove = []
for i in range(len(flux_source_ids)):
    if flux_source_ids[i] in flux_source_ids[:i]:
        duplicates_to_remove.append(i)
print(len(duplicates_to_remove))

np.savez('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz', source_ids=np.delete(
    flux_source_ids, duplicates_to_remove), fluxes=np.delete(d['fluxes'], duplicates_to_remove, axis=0))

with np.load('rvs_data_full_with_params4_highsnr.npz', allow_pickle=True) as d:
    source_ids = d['gaia_id']
    teff = d['teff']
    logg = d['logg']
    mh = d['mh']
    ag = d['ag']
    distance = d['distance']
    bprp_color = d['bprp_color']
    Mag = d['Mag']
    new_ang = d['new_ang']
    galactic_x = d['galactic_x']
    galactic_y = d['galactic_y']
    galactic_z = d['galactic_z']
    rv = d['rv']
    rv_err = d['rv_err']
    astrometric_excess_noise = d['astrometric_excess_noise']
    phot_g_mean_flux_over_error = d['phot_g_mean_flux_over_error']
    rv_expected_sig_to_noise = d['rv_expected_sig_to_noise']
    grvs_mag = d['grvs_mag']
    rvs_spec_sig_to_noise = d['rvs_spec_sig_to_noise']

np.savez('rvs_data_full_with_params4_highsnr_no_duplicates.npz', gaia_id=np.delete(source_ids, duplicates_to_remove), teff=np.delete(teff, duplicates_to_remove), logg=np.delete(logg, duplicates_to_remove), mh=np.delete(mh, duplicates_to_remove), ag=np.delete(ag, duplicates_to_remove), distance=np.delete(distance, duplicates_to_remove), bprp_color=np.delete(bprp_color, duplicates_to_remove), Mag=np.delete(Mag, duplicates_to_remove), new_ang=np.delete(new_ang, duplicates_to_remove), galactic_x=np.delete(galactic_x, duplicates_to_remove), galactic_y=np.delete(
    galactic_y, duplicates_to_remove), galactic_z=np.delete(galactic_z, duplicates_to_remove), rv=np.delete(rv, duplicates_to_remove), rv_err=np.delete(rv_err, duplicates_to_remove), astrometric_excess_noise=np.delete(astrometric_excess_noise, duplicates_to_remove), phot_g_mean_flux_over_error=np.delete(phot_g_mean_flux_over_error, duplicates_to_remove), rv_expected_sig_to_noise=np.delete(rv_expected_sig_to_noise, duplicates_to_remove), grvs_mag=np.delete(grvs_mag, duplicates_to_remove), rvs_spec_sig_to_noise=np.delete(rvs_spec_sig_to_noise, duplicates_to_remove))

with np.load('all_weirdness_scores_highsnr_no_nn.npz', allow_pickle=True) as d:
    weirdness_score = d['weirdness_score']
    source_ids_weirdness = d['source_ids']

np.savez('all_weirdness_scores_highsnr_no_nn_no_duplicates.npz', weirdness_score=np.delete(
    weirdness_score, duplicates_to_remove), source_ids=np.delete(source_ids_weirdness, duplicates_to_remove))

with np.load('umap_embedding_25nn_highsnr.npz', allow_pickle=True) as d:
    global X_embedded
    X_embedded = d['X_embedded']

np.savez('umap_embedding_25nn_highsnr_no_duplicates.npz',
         X_embedded=np.delete(X_embedded, duplicates_to_remove, axis=0))
