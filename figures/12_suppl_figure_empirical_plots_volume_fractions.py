import itertools
import subprocess
from textwrap import dedent

from scipy.stats import norm

import mdt
import matplotlib.pyplot as plt
from mdt.lib.post_processing import DTIMeasures
import numpy as np
import seaborn
seaborn.set()


__author__ = 'Robbert Harms'
__date__ = '2018-11-02'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


input_pjoin = mdt.make_path_joiner('/home/robbert/programming/python/uncertainty_paper/data/single_slice/')
output_base_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/single_slice/')
figure_output_pjoin = mdt.make_path_joiner('/tmp/uncertainty_paper/mean_mode_comparison')


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)


def get_mask(dataset_name):
    if dataset_name == 'mgh':
        mask_name = 'mgh_1003_slice_44_mask'
    else:
        mask_name = 'rheinland_v3a_1_ms20_slice_36_mask'
    return mdt.load_brain_mask(input_pjoin(dataset_name, mask_name))


def get_samples(dataset_name, model_name, voxel_volume_ind):
    mask = get_mask(dataset_name)
    roi_ind = mdt.volume_index_to_roi_index(voxel_volume_ind, mask)

    samples = mdt.load_samples(output_base_pjoin(dataset_name, model_name, 'samples'))

    if model_name == 'Tensor':
        return DTIMeasures.fractional_anisotropy(samples['Tensor.d'][roi_ind],
                                                 samples['Tensor.dperp0'][roi_ind],
                                                 samples['Tensor.dperp1'][roi_ind])
    elif model_name == 'BallStick_r1':
        return samples['w_stick0.w'][roi_ind]
    elif model_name == 'BallStick_r2':
        return samples['w_stick0.w'][roi_ind] + samples['w_stick1.w'][roi_ind]
    elif model_name == 'NODDI':
        return samples['w_ic.w'][roi_ind]
    elif model_name == 'CHARMED_r1':
        return samples['w_res0.w'][roi_ind]
    elif model_name == 'CHARMED_r2':
        return samples['w_res0.w'][roi_ind] + samples['w_res1.w'][roi_ind]
    elif model_name == 'BinghamNODDI_r1':
        return samples['w_in0.w'][roi_ind]


def get_mle_results(dataset_name, model_name):
    maps = mdt.load_volume_maps(output_base_pjoin(dataset_name, model_name))

    if model_name == 'Tensor':
        return maps['Tensor.FA'], maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return maps['w_stick0.w'], maps['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        return maps['FS'], maps['FS.std']
    elif model_name == 'NODDI':
        return maps['w_ic.w'], maps['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return maps['FR'], maps['FR.std']
    elif model_name == 'CHARMED_r2':
        return maps['FR'], maps['FR.std']
    elif model_name == 'BinghamNODDI_r1':
        return maps['w_in0.w'], maps['w_in0.w.std']


def get_mcmc_results(dataset_name, model_name):
    model_defined_maps = mdt.load_volume_maps(output_base_pjoin(
        dataset_name, model_name, 'samples', 'model_defined_maps'))
    univariate_normal_maps = mdt.load_volume_maps(output_base_pjoin(
        dataset_name, model_name, 'samples', 'univariate_normal'))

    if model_name == 'Tensor':
        return model_defined_maps['Tensor.FA'], model_defined_maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return model_defined_maps['FS'], model_defined_maps['FS.std']
    elif model_name == 'BallStick_r2':
        return model_defined_maps['FS'], model_defined_maps['FS.std']
    elif model_name == 'NODDI':
        return univariate_normal_maps['w_ic.w'], univariate_normal_maps['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return model_defined_maps['FR'], model_defined_maps['FR.std']
    elif model_name == 'CHARMED_r2':
        return model_defined_maps['FR'], model_defined_maps['FR.std']
    elif model_name == 'BinghamNODDI_r1':
        return univariate_normal_maps['w_in0.w'], univariate_normal_maps['w_in0.w.std']


def plot_histogram(ax, dataset_name, model_name, samples, mle, mle_std):
    ax.hist(samples, 100, density=True, color='gray', edgecolor='black')

    fit_x_coords = np.linspace(np.min(samples)*0.99, np.max(samples) * 1.01, 300)
    fit_y_coords_sampling = norm.pdf(fit_x_coords, loc=np.mean(samples), scale=np.std(samples))
    fit_y_coords_opt = norm.pdf(fit_x_coords, loc=mle, scale=mle_std)

    ax.plot(fit_x_coords, fit_y_coords_opt, '#d55e00', linewidth=4)
    ax.plot(fit_x_coords, fit_y_coords_sampling, '#56b4e9', linewidth=4)

    ax.plot(
        mle,
        # float(mlab.normpdf(mle, mle, mle_std)),
        float(norm.pdf(mle, mle, mle_std)),
        color='#d55e00', marker='v', markersize=18,
        label='MLE')

    ax.plot(
        np.mean(samples),
        # float(mlab.normpdf(np.mean(samples), np.mean(samples), np.std(samples))),
        float(norm.pdf(np.mean(samples), np.mean(samples), np.std(samples))),
        color='#56b4e9', marker='o', markersize=18,
        label='MCMC')

    ax.legend(loc='upper right')
    ax.set_xlabel(map_titles[model_name] + ' (a.u.)')
    ax.set_ylabel('Frequency (a.u.)')

    if ax.get_xlim()[1] > 1:
        ax.set_xlim(ax.get_xlim()[0], 1)

    if dataset_name == 'rheinland' and model_name == 'BinghamNODDI_r1':
        ax.set_xlim(np.mean(samples) - 0.09, np.mean(samples) + 0.09)
        ax.set_ylim(0, ax.get_ylim()[1])

    ax.get_yticklabels()[-1].set_verticalalignment('top')


def plot_maps(dataset_name, model_name, voxel_volume_ind, maps, out_name):
    scales = {'point_vmin': 0,
              'point_vmax': 1,
              'std_vmin': 0,
              'std_vmax': 0.04}

    if model_name == 'Tensor':
        scales['std_vmax'] = 0.15

    maps['diff'] = np.abs(maps['mcmc'] - maps['mle'])

    plot_config = dedent('''
        font: {{family: sans-serif, size: 20}}
        maps_to_show: [mle, mle_std, mcmc, mcmc_std]
        map_plot_options:
          mle:
            title: FS (MLE)
            scale: {{use_max: true, use_min: true, vmax: {point_vmax}, vmin: {point_vmin}}}
            colorbar_settings: {{round_precision: 2}}
          mle_std:
            title: FS std. (MLE)
            scale: {{use_max: true, use_min: true, vmax: {std_vmax}, vmin: {std_vmin}}}
            colorbar_settings: {{round_precision: 3}}
          mcmc:
            title: FS (MCMC)
            scale: {{use_max: true, use_min: true, vmax: {point_vmax}, vmin: {point_vmin}}}
            colorbar_settings: {{round_precision: 2}}
          mcmc_std:
            title: FS std. (MCMC)
            scale: {{use_max: true, use_min: true, vmax: {std_vmax}, vmin: {std_vmin}}}
            colorbar_settings: {{round_precision: 3}}
    '''.format(map_title=map_titles[model_name], **scales))

    if dataset_name == 'mgh':
        plot_config += dedent('''
            annotations:
            - arrow_width: 0.6
              font_size: 14
              marker_size: 3.0
              text_distance: 0.05
              text_location: top left
              text_template: '{value:.3f}'
              voxel_index: [''' + ', '.join(map(str, voxel_volume_ind)) + ''']
            colorbar_settings:
              location: right
              nmr_ticks: 4
              power_limits: [-3, 4]
              round_precision: 2
              visible: true
            grid_layout:
            - Rectangular
            - cols: null
              rows: null
              spacings: {bottom: 0.03, hspace: 0.0, left: 0.1, right: 0.86, top: 0.97, wspace: 0.5}
            rotate: 270
            zoom:
              p0: {x: 19, y: 17}
              p1: {x: 117, y: 128}
        ''')
    else:
        plot_config += dedent('''
            annotations:
            - arrow_width: 0.6
              font_size: 14
              marker_size: 3.0
              text_distance: 0.05
              text_location: top left
              text_template: '{value:.3f}'
              voxel_index: [''' + ', '.join(map(str, voxel_volume_ind)) + ''']
            colorbar_settings:
              location: right
              nmr_ticks: 4
              power_limits: [-3, 4]
              round_precision: 2
              visible: true
            grid_layout:
            - Rectangular
            - cols: null
              rows: null
              spacings: {bottom: 0.03, hspace: 0.0, left: 0.1, right: 0.86, top: 0.97, wspace: 0.5}
            zoom:
              p0: {x: 19, y: 6}
              p1: {x: 87, y: 97}
            title_spacing: 0.03
        ''')

    mdt.view_maps(
        maps,
        save_filename=out_name,
        figure_options={'width': 520, 'dpi': 80},
        config=plot_config)


def make_figure(dataset_name, model_name, voxel_volume_ind):
    samples = get_samples(dataset_name, model_name, voxel_volume_ind)
    mle, mle_std = get_mle_results(dataset_name, model_name)
    mcmc, mcmc_std = get_mcmc_results(dataset_name, model_name)

    img_output_pjoin = figure_output_pjoin.create_extended(
        '{}_{}_{}'.format(dataset_name, model_name, '_'.join(map(str, voxel_volume_ind))), make_dirs=True)

    set_matplotlib_font_size(18)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6.9, 7, forward=True)
    plot_histogram(ax, dataset_name, model_name, samples, mle[voxel_volume_ind], mle_std[voxel_volume_ind])
    # plt.show()
    fig.tight_layout()
    plt.savefig(img_output_pjoin('histogram.png'), dpi=80)

    skull = np.where(get_mle_results(dataset_name, 'BallStick_r1')[1] > 0.1)
    maps = {'mle': mle, 'mle_std': mle_std, 'mcmc': mcmc, 'mcmc_std': mcmc_std}
    for m in maps.values():
        m[skull] = 0

    plot_maps(dataset_name, model_name, voxel_volume_ind, maps, img_output_pjoin('maps.png'))

    subprocess.Popen('''
        convert maps.png histogram.png +append intro_figure.png
        convert intro_figure.png -splice 0x40 intro_figure.png
        convert intro_figure.png -font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \\
            -gravity center -pointsize 28 -fill '#282828' -annotate +0-300 '{model_title}' intro_figure.png
    '''.format(model_title=model_titles[model_name]), shell=True, cwd=img_output_pjoin()).wait()


map_titles = {
    'BallStick_r1': 'FS',
    'BallStick_r2': 'FS',
    'NODDI': 'FR',
    'BinghamNODDI_r1': 'FR',
    'Tensor': 'FA',
    'CHARMED_r1': 'FR',
    'CHARMED_r2': 'FR'
}


model_titles = {
    'BallStick_r1': 'BallStick_in1',
    'BallStick_r2': 'BallStick_in2',
    'BallStick_r3': 'BallStick_in3',
    'Tensor': 'Tensor',
    'NODDI': 'NODDI',
    'BinghamNODDI_r1': 'Bingham-NODDI',
    'CHARMED_r1': 'CHARMED_in1',
    'CHARMED_r2': 'CHARMED_in2',
    'CHARMED_r3': 'CHARMED_in3'
}

model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'NODDI',
    'BinghamNODDI_r1',
    'Tensor',
    'CHARMED_r1',
    'CHARMED_r2'
]

for model_name in model_names:
    make_figure('mgh', model_name, (68, 43, 0))
    make_figure('rheinland', model_name, (40, 79, 0))