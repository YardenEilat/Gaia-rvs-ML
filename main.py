# To run this app, use 'bokeh serve --show main.py' in the terminal

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from scipy.optimize import minimize
from bokeh.models import Button, MultiSelect, Select
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column
from bokeh.models import ColorBar, ColumnDataSource, LassoSelectTool, TapTool, Band
from bokeh.models.mappers import LinearColorMapper


def load_data():

    with np.load('rvs_data_full_with_params5_highsnr_no_duplicates.npz', allow_pickle=True) as d:
        global source_ids
        source_ids = d['gaia_id']
        teff = d['teff']
        logg = d['logg']
        mh = d['mh']
        ag = d['ag']
        global distance
        distance = d['distance']
        bprp_color = d['bprp_color']
        Mag = d['Mag']
        global new_ang
        new_ang = d['new_ang']
        global galactic_x
        galactic_x = d['galactic_x']
        global galactic_y
        galactic_y = d['galactic_y']
        global galactic_z
        galactic_z = d['galactic_z']
        rv = d['rv']
        rv_err = d['rv_err']
        astrometric_excess_noise = d['astrometric_excess_noise']
        phot_g_mean_flux_over_error = d['phot_g_mean_flux_over_error']
        rv_expected_sig_to_noise = d['rv_expected_sig_to_noise']
        grvs_mag = d['grvs_mag']
        rvs_spec_sig_to_noise = d['rvs_spec_sig_to_noise']
        rv_method = d['rv_method']

    with np.load('all_weirdness_scores_highsnr_no_nn_no_duplicates.npz', allow_pickle=True) as d:
        weirdness_score = d['weirdness_score']
        source_ids_weirdness = d['source_ids']
    assert np.all(source_ids == source_ids_weirdness)
    with np.load('umap_embedding_25nn_highsnr_no_duplicates.npz', allow_pickle=True) as d:
        global X_embedded
        X_embedded = d['X_embedded']
    astroparams = pd.read_csv(
        'astroparams_sorted_highsnr_no_duplicates.csv').to_dict(orient='list')
    global switcher
    switcher = {'Magnitudes': Mag, 'BP-RP color': bprp_color, 'Distance': distance,
                'M_H': mh, 'Teff': teff, 'logg': logg, 'alpha': ag, 'weirdness score': weirdness_score,
                'rv': rv, 'rv_err': rv_err, 'astrometric_excess_noise': astrometric_excess_noise,
                'phot_g_mean_flux_over_error': phot_g_mean_flux_over_error,
                'rv_expected_sig_to_noise': rv_expected_sig_to_noise, 'grvs_mag': grvs_mag, 'rvs_spec_sig_to_noise': rvs_spec_sig_to_noise, 'rv_method': rv_method}
    switcher = {**switcher, **astroparams}
    np.random.seed(42)
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


def get_source(x, y, total_selected_inds):
    global source
    global nan_source
    global nan_indices
    global not_nan_indices
    if any([type(item) is str for item in switcher[clr_dropdown.value]]):
        mapping = {}
        for i, obj_name in enumerate(list(set([obj for obj in list(switcher[clr_dropdown.value]) if obj == obj]))):
            mapping[obj_name] = i
        print(f'mapping: {mapping}')
        z = np.zeros(len(switcher[clr_dropdown.value]))
        for i, val in enumerate(switcher[clr_dropdown.value]):
            if val in mapping.keys():
                z[i] = mapping[val]
            else:
                z[i] = np.nan
    else:
        z = np.array(switcher[clr_dropdown.value], dtype=np.float64)
    if len(total_selected_inds) > 0:
        print(
            f'median value on selection: {np.nanmedian(z[total_selected_inds])}')
    nan_indices = np.where(np.isnan(z))[0]
    not_nan_indices = np.where(np.logical_not(np.isnan(z)))[0]
    z[nan_indices] = -1000
    source = ColumnDataSource(dict(x=x[not_nan_indices], y=y[not_nan_indices],
                              z=z[not_nan_indices], gaia_id=source_ids[not_nan_indices]))
    source.selected.indices = np.where(
        np.isin(not_nan_indices, total_selected_inds))[0]
    nan_source = ColumnDataSource(
        dict(x=x[nan_indices], y=y[nan_indices], gaia_id=source_ids[nan_indices]))
    nan_source.selected.indices = np.where(
        np.isin(nan_indices, total_selected_inds))[0]
    return source, z, nan_source, nan_indices, not_nan_indices


def plot_hr(total_selected_inds):
    fig = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_WIDTH)
    fig.add_tools(LassoSelectTool())
    fig.add_tools(TapTool())
    x = switcher['BP-RP color']
    y = switcher['Magnitudes']
    global source
    source, z, nan_source, nan_indices, not_nan_indices = get_source(
        x, y, total_selected_inds)
    from bokeh.palettes import Viridis256
    mapper = LinearColorMapper(palette=Viridis256, low=np.percentile(z[np.argwhere(
        z > -1000)], 5), high=np.percentile(z[np.argwhere(z > -1000)], 95), nan_color='gray')
    fig.circle(x='x', y='y', color='gray', size=2, source=nan_source)
    fig.circle(x='x', y='y', color={
               'field': 'z', 'transform': mapper}, size=2, source=source, alpha=0.7)
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
    # increase colorbar font size
    color_bar.major_label_text_font_size = '15pt'
    fig.add_layout(color_bar, 'right')
    fig.y_range.flipped = True
    fig.xaxis.axis_label = 'BP-RP color'
    fig.yaxis.axis_label = 'Magnitude'
    fig.title.text = f'CM Diagram - colored by {clr_dropdown.value}'
    curdoc().add_root(column(row(plt_dropdown, clr_dropdown, button,
                                 avg_spec_dropdown), row(fig, multi_select, save_button)))
    return total_selected_inds, nan_indices, not_nan_indices


def plot_galactic(total_selected_inds):
    fig = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_WIDTH, match_aspect=True)
    fig.add_tools(LassoSelectTool())
    fig.add_tools(TapTool())
    # x = distance * np.cos(new_ang)
    # y = distance * np.sin(new_ang)
    x = galactic_x
    y = galactic_y
    global source
    source, z, nan_source, nan_indices, not_nan_indices = get_source(
        x, y, total_selected_inds)
    from bokeh.palettes import Viridis256
    mapper = LinearColorMapper(palette=Viridis256, low=np.percentile(z[np.argwhere(
        z > -1000)], 5), high=np.percentile(z[np.argwhere(z > -1000)], 95), nan_color='gray')
    fig.circle(x='x', y='y', color='gray', size=2, source=nan_source)
    fig.circle(x='x', y='y', color={
               'field': 'z', 'transform': mapper}, size=2, source=source, alpha=0.7)
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
    # increase colorbar font size
    color_bar.major_label_text_font_size = '15pt'
    # increase colorbar width
    color_bar.width = 15
    fig.add_layout(color_bar, 'right')
    fig.xaxis.axis_label = 'Galactic x [kpc]'
    fig.yaxis.axis_label = 'Galactic y [kpc]'
    fig.title.text = f'Galactic plane - colored by {clr_dropdown.value}'
    curdoc().add_root(column(row(plt_dropdown, clr_dropdown, button,
                                 avg_spec_dropdown), row(fig, multi_select, save_button)))
    return total_selected_inds, nan_indices, not_nan_indices


def plot_tsne(total_selected_inds):
    global source
    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    source, z, nan_source, nan_indices, not_nan_indices = get_source(
        x, y, total_selected_inds)
    from bokeh.palettes import Viridis256
    mapper = LinearColorMapper(palette=Viridis256, low=np.percentile(z[np.argwhere(
        z > -1000)], 5), high=np.percentile(z[np.argwhere(z > -1000)], 95), nan_color='gray')
    fig = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_WIDTH,
                 tooltips=[("Apogee ID", "@apo_id"), ("GAIA ID", "@gaia_id"), ("value", "@z")])
    fig.add_tools(LassoSelectTool())
    fig.add_tools(TapTool())
    fig.circle(x='x', y='y', color='gray', size=2, source=nan_source)
    fig.circle(x='x', y='y', color={
               'field': 'z', 'transform': mapper}, size=2, source=source, alpha=0.7)
    fig.xaxis.axis_label = 'UMAP X'
    fig.yaxis.axis_label = 'UMAP Y'
    fig.title.text = f'UMAP projection - colored by {clr_dropdown.value}'
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
    # increase colorbar font size
    color_bar.major_label_text_font_size = '15pt'
    # increase colorbar width
    color_bar.width = 15
    fig.add_layout(color_bar, 'right')
    curdoc().add_root(column(row(plt_dropdown, clr_dropdown, button,
                                 avg_spec_dropdown), row(fig, column(multi_select, save_button))))
    return total_selected_inds, nan_indices, not_nan_indices


def plot_side_view(total_selected_inds):
    fig = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_WIDTH, match_aspect=True,
                 tooltips=[("Apogee ID", "@apo_id"), ("GAIA ID", "@gaia_id"), ("value", "@z")])
    fig.add_tools(LassoSelectTool())
    fig.add_tools(TapTool())
    x = galactic_x
    y = galactic_z
    global source
    source, z, nan_source, nan_indices, not_nan_indices = get_source(
        x, y, total_selected_inds)
    from bokeh.palettes import Viridis256
    mapper = LinearColorMapper(palette=Viridis256, low=np.percentile(z[np.argwhere(
        z > -1000)], 5), high=np.percentile(z[np.argwhere(z > -1000)], 95), nan_color='gray')

    fig.circle(x='x', y='y', color='gray', size=2, source=nan_source)
    fig.circle(x='x', y='y', color={
               'field': 'z', 'transform': mapper}, size=2, source=source, alpha=0.7)
    fig.xaxis.axis_label = 'Galactic x [kpc]'
    fig.yaxis.axis_label = 'Galactic z [kpc]'
    fig.title.text = f'Galactic side view - colored by {clr_dropdown.value}'
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0))
    fig.add_layout(color_bar, 'right')
    # increase colorbar font size
    color_bar.major_label_text_font_size = '15pt'
    # increase colorbar width
    color_bar.width = 15
    curdoc().add_root(column(row(plt_dropdown, clr_dropdown, button,
                                 avg_spec_dropdown), row(fig, multi_select, save_button)))
    return total_selected_inds, nan_indices, not_nan_indices


def replot_from_button(event):
    replot([], [], [])


def save_selection():
    global source
    global nan_source
    global nan_indices
    global not_nan_indices
    selected_inds = source.selected.indices
    selected_nan_inds = nan_source.selected.indices
    try:
        total_selected_inds = list(
            not_nan_indices[selected_inds]) + list(nan_indices[selected_nan_inds])
        if not total_selected_inds:
            total_selected_inds = []
    except:
        total_selected_inds = []
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    save_df = pd.DataFrame(data=dict(gaia_id=source_ids[total_selected_inds]))
    save_df.to_csv(f'save_{date}.csv', index=False)


def plot_parameter_histogram(total_selected_inds):
    """plots the histogram of the selected parameter for the selected sources"""
    if len(total_selected_inds) == 0:
        return
    param = np.array(switcher[clr_dropdown.value])[total_selected_inds]
    param = param[~np.isnan(param)]
    hist, edges = np.histogram(param)
    fig = figure(plot_width=PLOT_WIDTH, plot_height=400)
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
             fill_color="navy", line_color="white", alpha=0.5)
    fig.title.text = f'Histogram of {clr_dropdown.value} on selection. Median: {np.median(param)}'
    fig.xaxis.axis_label = f'{clr_dropdown.value}'
    fig.yaxis.axis_label = 'Count'
    curdoc().add_root(fig)


def replot(attr, old, new):
    print('CALCULATING...')
    curdoc().clear()
    global nan_indices
    global not_nan_indices
    global source
    global nan_source
    try:
        selected_inds = source.selected.indices
        print(selected_inds)
        selected_nan_inds = nan_source.selected.indices
        print(selected_nan_inds)
    except:
        selected_inds = []
        selected_nan_inds = []
    print(f'inds selected: {selected_inds}')
    total_selected_inds = list(
        not_nan_indices[selected_inds]) + list(nan_indices[selected_nan_inds])
    print(f'total inds selected: {total_selected_inds}')
    global multi_select
    multi_select = MultiSelect(value=[], options=list(source_ids[total_selected_inds].astype(str)), max_height=400,
                               sizing_mode="stretch_height")
    global save_button
    save_button = Button(
        label="Save selection to user data", button_type="success")
    save_button.on_click(save_selection)

    if plt_dropdown.value == 'UMAP':
        total_selected_inds, nan_indices, not_nan_indices = plot_tsne(
            total_selected_inds)
    elif plt_dropdown.value == 'Galactic plane':
        total_selected_inds, nan_indices, not_nan_indices = plot_galactic(
            total_selected_inds)
    elif plt_dropdown.value == 'CM Diagram':
        total_selected_inds, nan_indices, not_nan_indices = plot_hr(
            total_selected_inds)
    else:
        total_selected_inds, nan_indices, not_nan_indices = plot_side_view(
            total_selected_inds)
    plot_parameter_histogram(total_selected_inds)
    if 'Yes' in avg_spec_dropdown.value:
        plot_avg_spec(total_selected_inds)
    print('DONE')
    return source


def dropdown_handler(event):
    print(event)
    return


def L1_reg(X, y):

    def fit(X, params):
        return X * params[0] + params[1]

    def cost_function(params, X, y):
        return np.sum(np.abs(y - fit(X, params)))

    output = minimize(cost_function, (0, 1), args=(X, y))

    y_hat = fit(X, output.x)

    return output.x[0]


def plot_avg_spec(total_inds_selected):
    wavelengths = np.load('wavelength.npy')
    if len(total_inds_selected) == 0:
        return total_inds_selected
    elif len(total_inds_selected) == 1:
        d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                    allow_pickle=True, mmap_mode='r+')
        fluxes = d['fluxes'][indices][total_inds_selected, :]
        plt = figure(plot_width=PLOT_WIDTH, plot_height=400)
        plt.output_backend = "svg"
        plt.title.text = 'median RVS spectra on selection'
        plt.xaxis.axis_label = 'wavelength [nm]'
        plt.line(wavelengths,
                 np.ravel(fluxes), color='blue')
    else:
        d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz',
                    allow_pickle=True, mmap_mode='r+')
        total_fluxes = d['fluxes'][indices][total_inds_selected, :]
        fluxes = np.median(total_fluxes, axis=0)
        upper = np.percentile(total_fluxes, 90, axis=0)
        lower = np.percentile(total_fluxes, 10, axis=0)
        plt = figure(plot_width=PLOT_WIDTH, plot_height=400)
        plt.output_backend = "svg"
        plt.xaxis.axis_label = 'wavelength [nm]'
        plt.title.text = 'median RVS spectra on selection'
        # fix y limits
        plt.y_range.start = 0.3
        plt.y_range.end = 1.1
        print(fluxes)
        plt.line(wavelengths,
                 fluxes,
                 color='blue')
        band = Band(base='x', lower='lower', upper='upper', level='underlay', fill_alpha=0.5,
                    line_width=1, line_color='blue', source=ColumnDataSource({'x': wavelengths, 'lower': lower, 'upper': upper}))
        plt.add_layout(band)

        # add histogram of slope
        # from tqdm import tqdm
        # slope = [L1_reg(wavelengths, total_fluxes[i])
        #          for i in tqdm(range(len(total_inds_selected)))]
        # # convert slope to flux units per entire wavelength range
        # slope = np.array(slope) * (wavelengths[-1] - wavelengths[0])
        # plt2 = figure(plot_width=PLOT_WIDTH, plot_height=int(PLOT_HEIGHT/2))
        # plt2.output_backend = "svg"
        # plt2.xaxis.axis_label = 'flux slope'
        # plt2.yaxis.axis_label = 'count'
        # plt2.title.text = f'histogram of flux slopes (linear fit)'
        # hist, edges = np.histogram(slope, bins=50)
        # plt2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
        # curdoc().add_root(plt2)

        # add vertical lines for absorption lines
    if avg_spec_dropdown.value == 'Yes - with lines':
        line_wl = [{'NI': 863, 'NI2': 868.6, 'SiI': 853.9, 'CaII': 850, 'CaII2': 854.5, 'CaII3': 866.4,
                    'TiI': 857, 'CrI': 855, 'CrI2': 864.6, 'FeI': 857.4, 'FeI3': 858.5, 'FeI2': 862.4, 'FeII': 858.8, 'VO':862.4}]
        colors = ['red', 'green', 'yellow', 'orange', 'purple', 'pink',
                  'brown', 'black', 'grey', 'cyan', 'magenta', 'blue', 'olive', 'navy']
        for line in line_wl:
            for i, key in enumerate(line.keys()):
                plt.line([line[key], line[key]], [0, 1],
                         color=colors[i], legend_label=key)

    curdoc().add_root(plt)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent_to_display', type=int, default=10, 
                        help='percentage of objects to display. Choose a low number for a responsive experience, and 100 for presenting the full dataset.')
    parser.add_argument('--plot_width', type=int, default=800, 
                        help='width of the plots.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    PERCENT_TO_DISPLAY = args.percent_to_display
    PLOT_WIDTH = args.plot_width
    # START OF SCRIPT
    load_data()

    # Create the layout
    plt_type = ["UMAP", "Galactic plane", "Galactic side view", "CM Diagram"]
    plt_dropdown = Select(title='plot type:', value='UMAP', options=plt_type)
    plt_dropdown.on_change("value", replot)
    color_by = list(switcher.keys())
    clr_dropdown = Select(title="color by:", value="Magnitudes", options=color_by)
    clr_dropdown.on_change("value", replot)
    button = Button(label="Update Plots", button_type="success")
    avg_spec_dropdown = Select(title="plot spec?", value="No", options=[
                            'No', 'Yes', 'Yes - with lines'])

    # Create the initial source instance
    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    source = button.on_click(replot_from_button)
    source, z, nan_source, nan_indices, not_nan_indices = get_source(
        x, y, [])
    source = replot([], [], [])
