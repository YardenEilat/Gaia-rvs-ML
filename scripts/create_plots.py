import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_blue_giant_params():
    blue_giant_clust1_inds = pd.read_csv(
        'blue_giants_main_clust_inds.csv').values.astype(np.int64)
    blue_giant_clust1_inds = get_indices_from_ids(blue_giant_clust1_inds)
    blue_giant_clust2_inds = pd.read_csv(
        'blue_giants_separate_group_inds.csv').values.astype(np.int64)
    blue_giant_clust2_inds = get_indices_from_ids(blue_giant_clust2_inds)
    blue_giant_intermediate_inds = pd.read_csv(
        'blue_giants_inter_group_inds.csv').values.astype(np.int64)
    blue_giant_intermediate_inds = get_indices_from_ids(
        blue_giant_intermediate_inds)
    # plot histogram of effective temperature
    plt.figure()
    plt.hist(switcher['Teff'][blue_giant_clust1_inds],
             bins=np.linspace(np.nanpercentile(switcher['Teff'][blue_giant_clust1_inds], 0.5),
                              np.nanpercentile(switcher['Teff'][blue_giant_clust1_inds], 99), 100),
             color='blue',
             label='Grouped with MS stars', alpha=0.5)
    plt.hist(switcher['Teff'][blue_giant_clust2_inds],
             bins=np.linspace(np.nanpercentile(switcher['Teff'][blue_giant_clust2_inds], 0.5),
                              np.nanpercentile(switcher['Teff'][blue_giant_clust2_inds], 99), 100),
             color='green',
             label='Separate group', alpha=0.5)
    plt.hist(switcher['Teff'][blue_giant_intermediate_inds],
             bins=np.linspace(np.nanpercentile(switcher['Teff'][blue_giant_intermediate_inds], 0.5),
                              np.nanpercentile(switcher['Teff'][blue_giant_intermediate_inds], 99), 100),
             color='orange',
             label='Intermediate group', alpha=0.5)
    plt.xlabel('Effective temperature [K]')
    plt.ylabel('Number of objects')
    plt.title('Histogram of Effective Temperature of the blue giants')
    plt.legend()
    plt.savefig(os.path.join('umap_images_full',
                'blue_giants_teff_hist.svg'), dpi=600, bbox_inches='tight')

    # plot histogram of metallicity m_h
    plt.figure()
    plt.hist(switcher['M_H'][blue_giant_clust1_inds], bins=np.linspace(np.nanpercentile(switcher['M_H'][blue_giant_clust1_inds], 0.5),
             np.nanpercentile(switcher['M_H'][blue_giant_clust1_inds], 99), 100), color='blue', label='Grouped with MS stars', alpha=0.5)
    plt.hist(switcher['M_H'][blue_giant_clust2_inds], bins=np.linspace(np.nanpercentile(switcher['M_H'][blue_giant_clust2_inds], 0.5),
             np.nanpercentile(switcher['M_H'][blue_giant_clust2_inds], 99), 100), color='green', label='Separate group', alpha=0.5)
    plt.hist(switcher['M_H'][blue_giant_intermediate_inds], bins=np.linspace(np.nanpercentile(switcher['M_H'][blue_giant_intermediate_inds], 0.5),
             np.nanpercentile(switcher['M_H'][blue_giant_intermediate_inds], 99), 100), color='orange', label='Intermediate group', alpha=0.5)
    plt.xlabel('Metallicity [M/H]')
    plt.ylabel('Number of objects')
    plt.title('Histogram of Metallicity of the blue giants')
    plt.legend()
    plt.savefig(os.path.join('umap_images_full',
                'blue_giants_mh_hist.svg'), dpi=600, bbox_inches='tight')

    # plot histogram of teff_gspspec
    plt.figure()
    plt.hist(switcher['teff_gspspec'][blue_giant_clust1_inds],
             bins=np.linspace(np.nanpercentile(switcher['teff_gspspec'][blue_giant_clust1_inds], 0.5),
                              np.nanpercentile(switcher['teff_gspspec'][blue_giant_clust1_inds], 99), 100),
             color='blue',
             label='Grouped with MS stars', alpha=0.5)
    plt.hist(switcher['teff_gspspec'][blue_giant_clust2_inds],
             bins=np.linspace(np.nanpercentile(switcher['teff_gspspec'][blue_giant_clust2_inds], 0.5),
                              np.nanpercentile(switcher['teff_gspspec'][blue_giant_clust2_inds], 99), 100),
             color='green',
             label='Separate group', alpha=0.5)
    plt.hist(switcher['teff_gspspec'][blue_giant_intermediate_inds],
             bins=np.linspace(np.nanpercentile(switcher['teff_gspspec'][blue_giant_intermediate_inds], 0.5),
                              np.nanpercentile(switcher['teff_gspspec'][blue_giant_intermediate_inds], 99), 100),
             color='orange',
             label='Intermediate group', alpha=0.5)
    plt.xlabel('Effective temperature from gspspec [K]')
    plt.ylabel('Number of objects')
    plt.title('Histogram of Effective Temperature (gspspec) of the blue giants')
    plt.legend()
    plt.savefig(os.path.join('umap_images_full',
                'blue_giants_teff_gspspec_hist.svg'), dpi=600, bbox_inches='tight')
    # plot histogram of mh_gspspec
    plt.figure()
    plt.hist(switcher['mh_gspspec'][blue_giant_clust1_inds],
             bins=np.linspace(np.nanpercentile(switcher['mh_gspspec'][blue_giant_clust1_inds], 0.5),
                              np.nanpercentile(switcher['mh_gspspec'][blue_giant_clust1_inds], 99), 100),
             color='blue',
             label='Grouped with MS stars', alpha=0.5)
    plt.hist(switcher['mh_gspspec'][blue_giant_clust2_inds],
             bins=np.linspace(np.nanpercentile(switcher['mh_gspspec'][blue_giant_clust2_inds], 0.5),
                              np.nanpercentile(switcher['mh_gspspec'][blue_giant_clust2_inds], 99), 100),
             color='green',
             label='Separate group', alpha=0.5)
    plt.hist(switcher['mh_gspspec'][blue_giant_intermediate_inds],
             bins=np.linspace(np.nanpercentile(switcher['mh_gspspec'][blue_giant_intermediate_inds], 0.5),
                              np.nanpercentile(switcher['mh_gspspec'][blue_giant_intermediate_inds], 99), 100),
             color='orange',
             label='Intermediate group', alpha=0.5)
    plt.xlabel('Metallicity from gspspec [M/H]')
    plt.ylabel('Number of objects')
    plt.legend()
    plt.title('Histogram of Metallicity (gspspec) of the blue giants')
    plt.savefig(os.path.join('umap_images_full',
                'blue_giants_mh_gspspec_hist.svg'), dpi=600, bbox_inches='tight')


def create_plot(x, y, c, title, xlable, ylabel, clabel, selected_inds=None, flip_y_axis=False):
    if type(c) is str and c == 'slopes':
        # calculate slopes and set as c
        wavelengths = np.load('wavelength.npy')
        d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                    allow_pickle=True, mmap_mode='r+')
        fluxes = d['fluxes'][indices][selected_inds, :]
        c = [L1_reg(wavelengths, fluxes[i])
             for i in tqdm(range(len(selected_inds)))]
        c = np.array(c)
    else:
        c = c[selected_inds]
    plt.figure()
    colormap = plt.cm.get_cmap('viridis')
    colormap.set_bad('gray')
    # colormap.clim(np.percentile(c, 5), np.percentile(c, 95))
    if selected_inds is not None:
        not_selected = np.setdiff1d(np.arange(len(x)), selected_inds)
        plt.scatter(x[not_selected], y[not_selected], c='gray', s=0.1, cmap=colormap,
                    plotnonfinite=False, alpha=0.1, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
        plt.scatter(x[selected_inds], y[selected_inds], c=c, s=0.1, cmap=colormap, plotnonfinite=False,
                    alpha=0.9, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
    else:
        plt.scatter(x, y, c=c, s=0.1, cmap=colormap, plotnonfinite=False,
                    alpha=0.5, vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
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


def plot_hr_umap():
    main_seq_inds = pd.read_csv('main_seq_inds.csv')
    main_seq_inds = main_seq_inds['gaia_id'].values.astype(np.int64)
    # main_seq_inds = [np.where(source_ids == i)[0][0] for i in main_seq_inds[1:]]
    main_seq_inds = get_indices_from_ids(main_seq_inds)
    red_dwarf_inds = pd.read_csv('red_dwarf_inds.csv')
    red_dwarf_inds = red_dwarf_inds['gaia_id'].values
    # red_dwarf_inds = [np.where(source_ids == i)[0][0] for i in red_dwarf_inds]
    red_dwarf_inds = get_indices_from_ids(red_dwarf_inds)
    red_giant_inds = pd.read_csv('red_giant_inds.csv')
    red_giant_inds = red_giant_inds['gaia_id'].values
    # red_giant_inds = [np.where(source_ids == i)[0][0] for i in red_giant_inds]
    red_giant_inds = get_indices_from_ids(red_giant_inds)
    blue_giant_inds = pd.read_csv('blue_giant_inds.csv')
    blue_giant_inds = blue_giant_inds['gaia_id'].values
    # blue_giant_inds = [np.where(source_ids == i)[0][0] for i in blue_giant_inds]
    blue_giant_inds = get_indices_from_ids(blue_giant_inds)
    red_supergiant_inds = pd.read_csv('red_supergiant_inds.csv')
    red_supergiant_inds = red_supergiant_inds['gaia_id'].values
    # red_supergiant_inds = [np.where(source_ids == i)[0][0] for i in red_supergiant_inds]
    red_supergiant_inds = get_indices_from_ids(red_supergiant_inds)

    inds = {'main sequence': main_seq_inds, 'red dwarfs': red_dwarf_inds, 'red giants': red_giant_inds,
            'blue giants': blue_giant_inds, 'red supergiants': red_supergiant_inds}
    for key, value in inds.items():
        if key == 'blue giants':
            create_plot(X_embedded[:, 0], X_embedded[:, 1], switcher['Teff'],
                        f'(b) UMAP - {key}', 'UMAP X', 'UMAP Y', 'Effective temperature [K]', value)
            create_plot(switcher['BP-RP color'], switcher['Magnitudes'], switcher['Teff'],
                        f'(a) CMD diagram - {key}', 'BP-RP color [mag]', 'G-band magnitude [mag]', 'Effective temperature [K]', value, flip_y_axis=True)
        elif key == 'red dwarfs' or key == 'red supergiants':
            create_plot(X_embedded[:, 0], X_embedded[:, 1], 'slopes',
                        f'(b) UMAP - {key}', 'UMAP X', 'UMAP Y', 'Fit slope [flux units / $\AA$]', value)
            create_plot(switcher['BP-RP color'], switcher['Magnitudes'], 'slopes',
                        f'(a) CMD diagram - {key}', 'BP-RP color [mag]', 'G-band magnitude [mag]', 'Fit slope [flux units / $\AA$]', value, flip_y_axis=True)
        else:
            create_plot(X_embedded[:, 0], X_embedded[:, 1], switcher['Magnitudes'],
                        f'(b) UMAP - {key}', 'UMAP X', 'UMAP Y', 'G-band magnitude [mag]', value)
            create_plot(switcher['BP-RP color'], switcher['Magnitudes'], switcher['Magnitudes'],
                        f'(a) CMD diagram - {key}', 'BP-RP color [mag]', 'G-band magnitude [mag]', 'G-band magnitude [mag]', value, flip_y_axis=True)


def plot_umaps():
    colors_titles = {'(a) G-band magnitude [mag]': switcher['Magnitudes'], '(b) Effective temperature [K]': switcher['Teff'], '(c) BP-RP color [mag]': switcher['BP-RP color'], '(d) Distance [pc]': switcher['Distance'],
                     '(e) Surface gravity (log(g))': switcher['logg'], '(f) Metallicity [M/H]': switcher['M_H']}

    for key, value in colors_titles.items():
        create_plot(X_embedded[:, 0], X_embedded[:, 1],
                    value, key, None, None, None, None)
    # source_ids = set(source_ids)


def L1_reg(X, y):

    def fit(X, params):
        return X * params[0] + params[1]

    def cost_function(params, X, y):
        return np.sum(np.abs(y - fit(X, params)))

    output = minimize(cost_function, (0, 1), args=(X, y))

    y_hat = fit(X, output.x)

    return output.x[0]


def plot_compare_clusters(clust1_ind_path, clust2_ind_path, legend1=None, legend2=None, title=None):
    clust1_inds = pd.read_csv(clust1_ind_path)
    clust1_inds = clust1_inds['gaia_id'].values.astype(np.int64)
    inds = []
    for i in clust1_inds:
        temp = np.where(source_ids == i)[0]
        if len(temp) > 0:
            inds.append(temp[0])
    clust1_inds = np.array(inds)
    clust2_inds = pd.read_csv(clust2_ind_path)
    clust2_inds = clust2_inds['gaia_id'].values.astype(np.int64)
    inds = []
    for i in clust2_inds:
        temp = np.where(source_ids == i)[0]
        if len(temp) > 0:
            inds.append(temp[0])
    clust2_inds = np.array(inds)
    wavelengths = np.load('wavelength.npy')
    d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                allow_pickle=True, mmap_mode='r+')
    fluxes_clust1 = d['fluxes'][indices][clust1_inds, :]
    fluxes_clust2 = d['fluxes'][indices][clust2_inds, :]
    clust1_slope = [L1_reg(wavelengths, fluxes_clust1[i])
                    for i in tqdm(range(len(clust1_inds)))]
    # convert slope to flux units per entire wavelength range
    clust1_slope = np.array(clust1_slope) * (wavelengths[-1] - wavelengths[0])
    clust2_slope = [L1_reg(wavelengths, fluxes_clust2[i])
                    for i in tqdm(range(len(clust2_inds)))]
    # convert slope to flux units per entire wavelength range
    clust2_slope = np.array(clust2_slope) * (wavelengths[-1] - wavelengths[0])
    plt.figure()
    bins = np.linspace(np.min([np.min(clust1_slope), np.min(clust2_slope)]), np.max(
        [np.max(clust1_slope), np.max(clust2_slope)]), 100)
    plt.hist(clust1_slope, bins=bins, alpha=0.5, label=legend1)
    plt.hist(clust2_slope, bins=bins, alpha=0.5, label=legend2)
    plt.legend()
    plt.xlabel('Slope [flux units / $\AA$]')
    plt.ylabel('Number of objects')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('umap_images_full', title.replace(' ', '_').replace(
        '/', '') + '_with_title.svg'), dpi=600, bbox_inches='tight')
    # plot effective temperature histogram
    plt.figure()
    plt.hist(switcher['Teff'][clust1_inds], alpha=0.5, label=legend1, bins=np.linspace(np.nanpercentile(switcher['Teff'][np.concatenate([clust1_inds, clust2_inds])], 0.5),
                                                                                       np.nanpercentile(switcher['Teff'][np.concatenate([clust1_inds, clust2_inds])], 99), 100))
    plt.hist(switcher['Teff'][clust2_inds], alpha=0.5, label=legend2, bins=np.linspace(np.nanpercentile(switcher['Teff'][np.concatenate([clust1_inds, clust2_inds])], 0.5),
                                                                                       np.nanpercentile(switcher['Teff'][np.concatenate([clust1_inds, clust2_inds])], 99), 100))
    plt.legend()
    plt.xlabel('Effective temperature [K]')
    plt.ylabel('Number of objects')
    # plt.xlim(np.nanpercentile(switcher['Teff'][np.concatenate([clust1_inds, clust2_inds])], 0.5),
    #          np.nanpercentile(switcher['Teff'][np.concatenate([clust1_inds, clust2_inds])], 99))
    plt.title('Histogram of Effective Temperature of the two groups')
    plt.savefig(os.path.join('umap_images_full', title.replace(' ', '_').replace(
        '/', '') + '_with_title_teff.svg'), dpi=600, bbox_inches='tight')


def plot_spectra(inds, name, title=None):
    """Plots the median sepectra of the given indices"""
    wavelengths = np.load('wavelength.npy')
    flux_path = 'filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz'
    d = np.load(flux_path, allow_pickle=True,
                mmap_mode='r+')
    total_fluxes = d['fluxes'][indices][inds]
    fluxes = np.median(total_fluxes, axis=0)
    upper = np.percentile(total_fluxes, 90, axis=0)
    lower = np.percentile(total_fluxes, 10, axis=0)
    plt.figure(figsize=(6.4, 3.2))
    plt.plot(wavelengths, fluxes)
    plt.fill_between(wavelengths, lower, upper, alpha=0.5)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux [arbitrary units]')
    plt.grid()
    # place title inside the plot on the top left corner
    if title is not None:
        # plt.title(title, loc='center', fontsize=12, fontweight=0, color='black', fontdict={'family': 'monospace'})
        plt.text(0.02, 0.95, title, transform=plt.gca().transAxes, fontsize=12,
                 fontweight=0, color='black', fontdict={'family': 'monospace'})
    plt.savefig(os.path.join('umap_images',
                f'{name}.svg'), bbox_inches='tight')
    plt.close()


def plots_for_alejandra():
    red_dwarf_separate_clust = pd.read_csv(
        'red_dwarf_separate_cluster.csv').values.astype(np.int64)
    red_dwarf_separate_clust = get_indices_from_ids(red_dwarf_separate_clust)
    # plot 3 random spectra
    for i in range(3):
        ind = np.random.choice(red_dwarf_separate_clust)
        plot_spectra(
            [ind], f'spectra_for_alejandra_{source_ids[ind]}', title=f'spectra of object id {source_ids[ind]}')

    red_supergiant_separate_clust = pd.read_csv(
        'red_supergiant_separate_cluster_inds.csv').values.astype(np.int64)
    red_supergiant_separate_clust = get_indices_from_ids(
        red_supergiant_separate_clust)
    # plot 3 random spectra
    for i in range(3):
        ind = np.random.choice(red_supergiant_separate_clust)
        plot_spectra(
            [ind], f'spectra_for_alejandra_{source_ids[ind]}', title=f'spectra of object id {source_ids[ind]}')


def plot_all_required_spectra():
    blue_giant_clust1_inds = pd.read_csv(
        'blue_giants_main_clust_inds.csv').values.astype(np.int64)
    blue_giant_clust1_inds = get_indices_from_ids(blue_giant_clust1_inds)
    plot_spectra(blue_giant_clust1_inds, 'blue_giants_main_clust_spect')
    blue_giant_clust2_inds = pd.read_csv(
        'blue_giants_separate_group_inds.csv').values.astype(np.int64)
    blue_giant_clust2_inds = get_indices_from_ids(blue_giant_clust2_inds)
    plot_spectra(blue_giant_clust2_inds, 'blue_giants_separate_group_spect',
                 title='(c) Spectra of different groups')
    blue_giant_intermediate_inds = pd.read_csv(
        'blue_giants_inter_group_inds.csv').values.astype(np.int64)
    blue_giant_intermediate_inds = get_indices_from_ids(
        blue_giant_intermediate_inds)
    plot_spectra(blue_giant_intermediate_inds,
                 'blue_giants_intermediate_group_spect')
    red_dwarf_main_clust = pd.read_csv(
        'red_dwarf_main_cluster.csv').values.astype(np.int64)
    red_dwarf_main_clust = get_indices_from_ids(red_dwarf_main_clust)
    plot_spectra(red_dwarf_main_clust, 'red_dwarfs_main_clust_spect',
                 title='(c) Spectra of different groups')
    red_dwarf_separate_clust = pd.read_csv(
        'red_dwarf_separate_cluster.csv').values.astype(np.int64)
    red_dwarf_separate_clust = get_indices_from_ids(red_dwarf_separate_clust)
    plot_spectra(red_dwarf_separate_clust, 'red_dwarfs_separate_clust_spect')
    red_supergiant_main_clust = pd.read_csv(
        'red_supergiant_main_cluster_inds.csv').values.astype(np.int64)
    red_supergiant_main_clust = get_indices_from_ids(
        red_supergiant_main_clust)
    plot_spectra(red_supergiant_main_clust,
                 'red_supergiants_main_clust_spect', title='(c) Spectra of different groups')
    red_supergiant_separate_clust = pd.read_csv(
        'red_supergiant_separate_cluster_inds.csv').values.astype(np.int64)
    red_supergiant_separate_clust = get_indices_from_ids(
        red_supergiant_separate_clust)
    plot_spectra(red_supergiant_separate_clust,
                 'red_supergiants_separate_clust_spect')


def get_leaf_inds():
    """loads RF model and returns leaf indices for each source"""
    import joblib
    d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                allow_pickle=True, mmap_mode='r+')
    fluxes = d['fluxes'][indices]
    rf_model = joblib.load('rvs_trained_rf_numtrees_500_highsnr.joblib')
    leaves = rf_model.apply(fluxes)
    return leaves


def get_indices_from_ids(ids):
    """Returns the indices of the given ids in the source_ids array"""
    inds = []
    for i in ids:
        temp = np.argmin(np.abs(source_ids - i))
        if np.abs(source_ids[temp] - i) < 1000:
            inds.append(temp)
        else:
            print(f'Could not find {i}')
    return np.array(inds)


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
astroparams = pd.read_csv(
    'astroparams_sorted_highsnr_no_duplicates.csv').to_dict(orient='list')
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

source_ids = source_ids[indices]
X_embedded = X_embedded[indices]
new_ang = new_ang[indices]
distance = distance[indices]
galactic_x = galactic_x[indices]
galactic_y = galactic_y[indices]
galactic_z = galactic_z[indices]
for key in switcher.keys():
    switcher[key] = np.array(switcher[key])[indices]

# plot_umaps()
# plot_hr_umap()
# plot_compare_clusters('red_dwarf_main_cluster.csv', 'red_dwarf_separate_cluster.csv', legend1='red dwarfs grouped with MS stars',
#                       legend2='red dwarfs in separate group', title='Histograms of fit slopes for the red dwarf groups')
# plot_compare_clusters('red_supergiant_main_cluster_inds.csv', 'red_supergiant_separate_cluster_inds.csv', legend1='red supergiants grouped with MS stars',
#                       legend2='red supergiants in separate group', title='Histograms of fit slopes for the red supergiant groups')
# plot_all_required_spectra()
# plots_for_alejandra()
# plot_blue_giant_params()
# # investigate weirdness spike
# spike_inds = pd.read_csv('spike_group_inds.csv')
# spike_inds = np.array(spike_inds)
# inds = []
# for i in spike_inds:
#     temp = np.where(source_ids == i)[0]
#     if len(temp) > 0:
#         inds.append(temp[0])
# spike_inds = np.array(inds)
# print(len(spike_inds))
# # create_plot(X_embedded[:,0], X_embedded[:,1], switcher['Magnitudes'], 'Spike group selected on UMAP', None, None, 'G-band magnitude [mag]', spike_inds)
# leaves = get_leaf_inds()
# leaves = leaves[spike_inds]
# leaves = leaves.T
# plt.figure()
# plt.hist(np.mean(leaves, axis=0))
# plt.show()
# print(leaves)
