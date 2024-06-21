import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_hists_compare(index, second_indices=None):
    """ compare index object to the rest of the sample """
    property_list = ['Magnitudes', 'BP-RP color',
                     'Distance', 'M_H', 'Teff', 'logg', 'alpha', 'rv', 'rv_err', 'astrometric_excess_noise',
                     'phot_g_mean_flux_over_error', 'rv_expected_sig_to_noise', 'grvs_mag', 'rvs_spec_sig_to_noise',
                     'teff_gspspec', 'logg_gspspec', 'mh_gspspec']
    i = 1
    plt.figure(figsize=(6.4, 3.2 * len(property_list)))
    for key in switcher.keys():
        if key not in property_list:
            continue
        plt.subplot(len(property_list), 1,  i,
                    title=f'{key} value = {switcher[key][index]}')
        plt.hist(switcher[key], bins=100, alpha=0.5, label=key, density=True)
        # Adjust x-axis limits
        plt.xlim(np.nanmin(switcher[key]), np.nanmax(switcher[key]))
        plt.axvline(switcher[key][index], color='red',
                    linestyle='dashed', linewidth=2)
        if second_indices is not None:
            plt.hist(switcher[key][second_indices], bins=100,
                     alpha=0.5, label=key, color='orange', density=True)
        i += 1
    plt.legend()
    plt.savefig(os.path.join('umap_images', 'hists_compare.svg'),
                bbox_inches='tight')
    plt.close()


def plot_spectra(inds, name, title=None, hold_on=False, fig=None, type='median', legend=None, add_lines=None):
    """Plots the median sepectra of the given indices"""
    wavelengths = np.load('wavelength.npy')
    flux_path = 'filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz'
    d = np.load(flux_path, allow_pickle=True,
                mmap_mode='r+')
    total_fluxes = d['fluxes'][indices][inds]
    if type=='median':
        fluxes = np.median(total_fluxes, axis=0)
        upper = np.percentile(total_fluxes, 90, axis=0)
        lower = np.percentile(total_fluxes, 10, axis=0)
        if not fig:
            plt.figure(figsize=(6.4, 3.2))
            plt.plot(wavelengths, fluxes)
            plt.fill_between(wavelengths, lower, upper, alpha=0.7)
        if fig:
            plt.plot(wavelengths, fluxes, alpha=0.5)
            plt.fill_between(wavelengths, lower, upper, alpha=0.45)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Flux [arbitrary units]')
        plt.grid()
    elif type=='offset':
        if not fig:
            plt.figure(figsize=(6.4, 6.4))
        for i in range(len(total_fluxes)):
            offset = i * 0.5
            fluxes = total_fluxes[i] - offset
            if not fig:
                plt.plot(wavelengths, fluxes, label=legend[i])
            if fig:
                plt.plot(wavelengths, fluxes, alpha=0.5, label=legend[i])
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Flux [arbitrary units] + offset')
        plt.yticks([])
        plt.legend(loc='upper right')
        plt.grid()
    else:
        raise ValueError('type must be either "median" or "offset"')
    if add_lines is not None:
        for line in add_lines:
            plt.axvline(line, color='red', linestyle=':', linewidth=2, alpha=0.5)
    # add VO line
    # plt.axvline(862, color='red', linestyle='dashed', linewidth=2)
    # place title inside the plot on the top left corner
    if title is not None:
        # plt.title(title, loc='center', fontsize=12, fontweight=0, color='black', fontdict={'family': 'monospace'})
        plt.text(0.02, 0.95, title, transform=plt.gca().transAxes, fontsize=12,
                 fontweight=0, color='black', fontdict={'family': 'monospace'})
    if not hold_on:
        plt.savefig(os.path.join('umap_images',
                    f'{name}.svg'), bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()


def create_plot(x, y, c, title, xlable, ylabel, clabel, selected_inds=None, second_selected_inds=None, flip_y_axis=False, numbers=False):
    if c == 'slopes':
        # calculate slopes and set as c
        wavelengths = np.load('wavelength.npy')
        d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                    allow_pickle=True, mmap_mode='r+')
        fluxes = d['fluxes'][indices][selected_inds, :]
        c = [L1_reg(wavelengths, fluxes[i])
             for i in tqdm(range(len(selected_inds)))]
        c = np.array(c)
    plt.figure()
    colormap = plt.cm.get_cmap('viridis')
    colormap.set_bad('gray')
    # colormap.clim(np.percentile(c, 5), np.percentile(c, 95))
    if selected_inds is not None:
        not_selected = np.setdiff1d(np.arange(len(x)), selected_inds)
        if c is not None:
            plt.scatter(x[not_selected], y[not_selected], c='gray', s=0.1, cmap=colormap,
                    plotnonfinite=False, alpha=0.1, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
            plt.scatter(x[selected_inds], y[selected_inds], c=c[selected_inds], s=5, cmap=colormap, plotnonfinite=False,
                        alpha=0.9, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
        else:
            plt.scatter(x[not_selected], y[not_selected], c='gray', s=0.1, alpha=0.1)
            plt.scatter(x[selected_inds], y[selected_inds], c='red', s=5)
        if numbers:
            for i, txt in enumerate(selected_inds):
                plt.text(x[selected_inds][i], y[selected_inds][i], i+1, fontsize=6)
    else:
        plt.scatter(x, y, c=c, s=0.1, cmap=colormap, plotnonfinite=False,
                    alpha=0.5, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
    if second_selected_inds is not None:
        plt.scatter(x[second_selected_inds], y[second_selected_inds], c=c[second_selected_inds], s=5, cmap=colormap,
                    plotnonfinite=False, alpha=0.5, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95), marker='x')
    if c is not None:
        cbar = plt.colorbar()
    # place title inside the plot on the top left corner
    if title is not None:
        # plt.title(title, loc='center', fontsize=12, fontweight=0, color='black', fontdict={'family': 'monospace'})
        plt.text(0.02, 0.95, title, transform=plt.gca().transAxes, fontsize=12,
                 fontweight=0, color='black', fontdict={'family': 'monospace'})
    if xlable is not None:
        plt.xlabel(xlable)
    else:   # don't show axis
        plt.xticks([])
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:   # don't show axis
        plt.yticks([])
    if clabel is not None:
        cbar.set_label(clabel)
    if flip_y_axis:
        plt.gca().invert_yaxis()

    # ax = plt.gca()
    # axins = zoomed_inset_axes(ax, 7, loc=3)
    # axins.scatter(x[not_selected], y[not_selected], c='gray', s=0.1, cmap=colormap,
    #               plotnonfinite=False, alpha=0.1, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
    # axins.scatter(x[selected_inds], y[selected_inds], c=c[selected_inds], s=0.1, cmap=colormap, plotnonfinite=False,
    #               alpha=0.9, vmin=np.nanpercentile(c[selected_inds], 5), vmax=np.nanpercentile(c[selected_inds], 95))
    # # sub region of the original image
    # x1, x2, y1, y2 = -29, -27, 13, 14
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticks([])
    # axins.set_yticks([])
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # draw a bbox of the region of the inset axes in the parent axes and
    # # connecting lines between the bbox and the inset axes area
    # mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    # save figure
    plt.savefig(os.path.join('umap_images', title.replace(' ', '_').replace(
        '/', '') + '_with_title.png'), dpi=600, bbox_inches='tight')


def get_closest_inds(index, n=10, method='hamming'):
    """Get the n closest indices to the given index"""
    if method == 'hamming':
        leaves = get_leaf_inds()
        hamming = [[0 if leaf == leaves[index, i] else 1 for i,
                    leaf in enumerate(leafs)] for leafs in leaves]
        distances = np.linalg.norm(hamming, axis=1)
    elif method == 'umap':
        distances = np.linalg.norm(
            X_embedded - X_embedded[index], axis=1)
    else:
        raise ValueError('method must be either "hamming" or "umap"')
    return np.argsort(distances)[1:n + 1]


def analyze_index(index):
    # print the flag value
    print(f'flag value is {switcher["flags_gspspec"][index]}')
    # print source id
    print(f'source id is {source_ids[index]}')
    # print actual source id
    df = pd.read_csv('gaia_params_lite.csv')
    df = df.iloc[np.argmin(np.abs(df['source_id'] - source_ids[index]))]
    print(f'actual source_id is {df["source_id"]}')
    closest_inds = get_closest_inds(index, n=5)
    # import random
    # random.seed(42)
    # closest_inds = random.sample(range(len(source_ids)), 10000)
    create_plot(X_embedded[:, 0], X_embedded[:, 1], switcher['Magnitudes'],
                'Weirdest object', None, None, None, selected_inds=[index], second_selected_inds=closest_inds)
    fig = plot_spectra([index], 'weirdest_object_spectra',
                       title=None, hold_on=True)
    plot_spectra(closest_inds, 'total_median_spectra',
                 title="blue - weird object. orange - total median spectra", fig=fig)
    # plot_hists_compare(index, closest_inds)



def get_leaf_inds():
    """loads RF model and returns leaf indices for each source"""
    import joblib
    d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                allow_pickle=True, mmap_mode='r+')
    fluxes = d['fluxes'][indices]
    rf_model = joblib.load('rvs_trained_rf_numtrees_500_highsnr.joblib')
    leaves = rf_model.apply(fluxes)
    return leaves


def plot_top_weirdest_umap(n=20):
    """Plots the n weirdest objects in the UMAP space"""
    weirdest_inds = np.argsort(switcher['weirdness score'])[-n:]
    weirdest_inds = weirdest_inds[::-1]
    create_plot(X_embedded[:, 0], X_embedded[:, 1], None,
                'Weirdest objects umap', None, None, None, selected_inds=weirdest_inds, numbers=True)
    
    
def analyze_pair(ind1, ind2):
    """For the two objects, determine what order of neighbors they are to each other"""
    leaves = get_leaf_inds()
    hamming = [[0 if leaf == leaves[ind1, i] else 1 for i,
                leaf in enumerate(leafs)] for leafs in leaves]
    distances = np.linalg.norm(hamming, axis=1)
    ind1_neighbors = np.argsort(distances)
    hamming = [[0 if leaf == leaves[ind2, i] else 1 for i,
                leaf in enumerate(leafs)] for leafs in leaves]
    distances = np.linalg.norm(hamming, axis=1)
    ind2_neighbors = np.argsort(distances)
    print(f'ind1 is {np.where(ind1_neighbors == ind2)[0][0]}th neighbor of ind2')
    print(f'ind2 is {np.where(ind2_neighbors == ind1)[0][0]}th neighbor of ind1')


if __name__ == "__main__":
    np.random.seed(42)
    PERCENT_TO_DISPLAY = 100

    with np.load('rvs_data_full_with_params5_highsnr_no_duplicates.npz', allow_pickle=True) as d:
        source_ids = d['gaia_id'].astype(np.int64)
        # weirdness_score = d['weirdness_score']
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

with np.load('all_weirdness_scores_highsnr_no_nn_no_duplicates.npz', allow_pickle=True) as d:
    weirdness_score = d['weirdness_score']
    source_ids_weirdness = d['source_ids']

with np.load('umap_embedding_25nn_highsnr_no_duplicates.npz', allow_pickle=True) as d:
    X_embedded = d['X_embedded']
astroparams = pd.read_csv('astroparams_sorted_highsnr_no_duplicates.csv')
astroparams = dict(astroparams)
switcher = {'Magnitudes': Mag, 'BP-RP color': bprp_color, 'Distance': distance,
            'M_H': mh, 'Teff': teff, 'logg': logg, 'alpha': ag, 'weirdness score': weirdness_score,
            'rv': rv, 'rv_err': rv_err, 'astrometric_excess_noise': astrometric_excess_noise,
            'phot_g_mean_flux_over_error': phot_g_mean_flux_over_error,
            'rv_expected_sig_to_noise': rv_expected_sig_to_noise, 'grvs_mag': grvs_mag, 'rvs_spec_sig_to_noise': rvs_spec_sig_to_noise}
switcher = {**switcher, **astroparams}
# create weirdness score histogram
# plt.figure()
# plt.hist(weirdness_score, bins=1000)
# plt.xlabel('Weirdness score')
# plt.ylabel('Count')
# plt.title('Weirdness score histogram')
# plt.savefig('weirdscore_hist.svg', dpi=600, bbox_inches='tight')
global indices
indices = np.random.choice(np.arange(len(source_ids)), int(PERCENT_TO_DISPLAY / 100 * len(source_ids)),
                           replace=False)
# filter indices with rvs_spec_sig_to_noise < 50
indices = indices[rvs_spec_sig_to_noise[indices] > 50]
source_ids = source_ids[indices]
X_embedded = X_embedded[indices]
new_ang = new_ang[indices]
distance = distance[indices]
galactic_x = galactic_x[indices]
galactic_y = galactic_y[indices]
galactic_z = galactic_z[indices]
for key in switcher.keys():
    switcher[key] = switcher[key][indices]
# analyze_index(np.argsort(switcher['weirdness score'])[-5])
analyze_pair(np.argsort(switcher['weirdness score'])[-4], np.argsort(switcher['weirdness score'])[-3])
# plot_top_weirdest_umap()
# plot MIRAs
# plot_spectra(np.argsort(switcher['weirdness score'])[list([-1, -3, -4])], 'MIRA type variables', title='MIRA type variables', 
#              type='offset', legend=['object 1', 'object 3', 'object 4'], add_lines=[850, 854.5, 866.4])
# plot S-type
# plot_spectra(np.argsort(switcher['weirdness score'])[list([-2, -14, -16, -19, -20])], 'S-type stars', title='S-type stars', 
#              type='offset', legend=['object 2', 'object 14', 'object 16', 'object 19', 'object 20 (cepheid)'])
# plot cool carbon
# plot_spectra(np.argsort(switcher['weirdness score'])[list([-17, -18])], 'cool carbon stars', title='cool carbon stars', 
#              type='offset', legend=['object 17', 'object 18'])
# plot yellowe and blue supergoiants
# plot_spectra(np.argsort(switcher['weirdness score'])[list([-5, -7, -12])], 'yellow and blue supergiant stars', title='yellow and blue supergiant stars', 
#              type='offset', legend=['object 5', 'object 7', 'object 12'])
# plot unknown
# plot_spectra(np.argsort(switcher['weirdness score'])[list([-6, -8, -10, -11, -13, -15])], 'unknown stars', title='unknown stars', 
#              type='offset', legend=['object 6', 'object 8', 'object 10', 'object 11', 'object 13', 'object 15'])