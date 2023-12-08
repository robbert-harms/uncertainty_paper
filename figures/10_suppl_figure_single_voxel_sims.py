import itertools
import subprocess
from textwrap import dedent

from matplotlib import mlab
from scipy.stats import norm

import mdt
import matplotlib.pyplot as plt
from mdt.lib.post_processing import DTIMeasures
import numpy as np
import seaborn

from mdt.utils import create_covariance_matrix

seaborn.set()


__author__ = 'Robbert Harms'
__date__ = '2018-11-02'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


input_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/single_voxel_simulation/')
figure_output_pjoin = mdt.make_path_joiner('/tmp/uncertainty_paper/single_voxel_simulations/')
figure_output_pjoin.make_dirs()


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)


def get_samples(base_pjoin):
    samples = mdt.load_samples(base_pjoin('samples'))

    if model_name == 'Tensor':
        return DTIMeasures.fractional_anisotropy(samples['Tensor.d'],
                                                 samples['Tensor.dperp0'],
                                                 samples['Tensor.dperp1'])
    elif model_name == 'BallStick_r1':
        return samples['w_stick0.w']
    elif model_name == 'BallStick_r2':
        return samples['w_stick0.w'] + samples['w_stick1.w']
    elif model_name == 'BallStick_r3':
        return samples['w_stick0.w'] + samples['w_stick1.w'] + samples['w_stick2.w']
    elif model_name == 'NODDI':
        return samples['w_ic.w']
    elif model_name == 'CHARMED_r1':
        return samples['w_res0.w']
    elif model_name == 'CHARMED_r2':
        return samples['w_res0.w'] + samples['w_res1.w']
    elif model_name == 'BinghamNODDI_r1':
        return samples['w_in0.w']


def get_mle_results(base_pjoin):
    maps = mdt.load_volume_maps(base_pjoin())

    if model_name == 'Tensor':
        return maps['Tensor.FA'], maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return maps['w_stick0.w'], maps['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        return maps['FS'], maps['FS.std']
    elif model_name == 'BallStick_r3':
        return maps['FS'], maps['FS.std']
    elif model_name == 'NODDI':
        return maps['w_ic.w'], maps['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return maps['FR'], maps['FR.std']
    elif model_name == 'CHARMED_r2':
        return maps['FR'], maps['FR.std']
    elif model_name == 'BinghamNODDI_r1':
        return maps['w_in0.w'], maps['w_in0.w.std']


def std_volume_fractions(weight_names, point_estimate, fim):
    results = {}
    results.update(point_estimate)
    results.update(fim)

    nmr_voxels = point_estimate[weight_names[1]].shape[0]
    covar_matrix = create_covariance_matrix(nmr_voxels, results, weight_names, results)
    covar_sum = np.sum(np.sum(covar_matrix, axis=1), axis=1)
    covar_sum[np.isinf(covar_sum) | np.isnan(covar_sum) | (covar_sum < 0)] = 0
    std = np.sqrt(covar_sum)
    return std


def get_crlb_results(base_pjoin, model_name, snr):
    free_param_names = mdt.get_model(model_name)().get_free_param_names()
    original_parameters = mdt.load_volume_maps(base_pjoin())['original_parameters']

    params = dict(zip(free_param_names, np.split(original_parameters, len(free_param_names), axis=3)))
    crlbs = mdt.load_volume_maps(base_pjoin('output', str(snr), 'CRLB', model_name, 'FIM'))

    if model_name == 'Tensor':
        fa = DTIMeasures.fractional_anisotropy(params['Tensor.d'],
                                               params['Tensor.dperp0'],
                                               params['Tensor.dperp1'])
        fa_std = DTIMeasures.fractional_anisotropy_std(
            params['Tensor.d'],
            params['Tensor.dperp0'],
            params['Tensor.dperp1'],
            crlbs['Tensor.d.std'],
            crlbs['Tensor.dperp0.std'],
            crlbs['Tensor.dperp1.std'], covariances=crlbs)
        return fa, fa_std
    elif model_name == 'BallStick_r1':
        return params['w_stick0.w'], crlbs['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        return params['w_stick0.w'] + params['w_stick1.w'], \
               std_volume_fractions(['w_stick0.w', 'w_stick1.w'], params, crlbs)
    elif model_name == 'BallStick_r3':
        return params['w_stick0.w'] + params['w_stick1.w'] + params['w_stick2.w'], \
               std_volume_fractions(['w_stick0.w', 'w_stick1.w', 'w_stick2.w'], params, crlbs)
    elif model_name == 'NODDI':
        return params['w_ic.w'], crlbs['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return params['w_res0.w'], crlbs['w_res0.w.std']
    elif model_name == 'CHARMED_r2':
        return params['w_res0.w'] + params['w_res1.w'], \
               std_volume_fractions(['w_res0.w', 'w_res1.w', 'w_stick2.w'], params, crlbs)
    elif model_name == 'BinghamNODDI_r1':
        return params['w_in0.w'], crlbs['w_in0.w.std']


def plot_histogram(ax, dataset_name, model_name, samples, mle, mle_std, gt_param, gt_crlb):
    ax.hist(samples, 100, density=True, color='gray', edgecolor='black')

    fit_x_coords = np.linspace(np.min(samples)*0.99, np.max(samples) * 1.01, 300)
    fit_y_coords_sampling = norm.pdf(fit_x_coords, loc=np.mean(samples), scale=np.std(samples))
    fit_y_coords_opt = norm.pdf(fit_x_coords, loc=mle, scale=mle_std)
    fit_y_coords_crlb = norm.pdf(fit_x_coords, loc=gt_param, scale=gt_crlb)

    # ax.plot(fit_x_coords, fit_y_coords_crlb, '#2e5e00', linewidth=4)
    ax.plot(fit_x_coords, fit_y_coords_opt, '#d55e00', linewidth=4)
    ax.plot(fit_x_coords, fit_y_coords_sampling, '#56b4e9', linewidth=4)

    # ax.plot(
    #     gt_param,
    #     float(norm.pdf(gt_param, gt_param, gt_crlb)),
    #     color='#2e5e00', marker='s', markersize=18,
    #     label='CRLB')

    ax.plot(
        mle,
        float(norm.pdf(mle, mle, mle_std)),
        color='#d55e00', marker='v', markersize=18,
        label='MLE')

    ax.plot(
        np.mean(samples),
        float(norm.pdf(np.mean(samples), np.mean(samples), np.std(samples))),
        color='#56b4e9', marker='o', markersize=18,
        label='MCMC')

    ax.legend(loc='upper right')
    # ax.set_xlabel(map_titles[model_name] + ' (a.u.)')
    ax.set_xlabel('Parameter of interest (a.u.)')
    ax.set_ylabel('Frequency (a.u.)')

    if ax.get_xlim()[1] > 1:
        ax.set_xlim(ax.get_xlim()[0], 1)

    if dataset_name == 'rheinland' and model_name == 'BinghamNODDI_r1':
        ax.set_xlim(np.mean(samples) - 0.09, np.mean(samples) + 0.09)
        ax.set_ylim(0, ax.get_ylim()[1])

    ax.get_yticklabels()[-1].set_verticalalignment('top')


def make_figure(dataset_name, model_name, snr):
    base_pjoin = input_pjoin.create_extended(dataset_name, model_name, 'output', str(snr), model_name)

    samples = get_samples(base_pjoin)
    mle, mle_std = get_mle_results(base_pjoin)
    original_parameter, crlb = get_crlb_results(input_pjoin.create_extended(dataset_name, model_name), model_name, snr)

    set_matplotlib_font_size(18)
    fig, ax = plt.subplots(1, 1)
    # fig.set_size_inches(6.9, 7, forward=True)
    fig.set_size_inches(8, 6, forward=True)
    plot_histogram(ax, dataset_name, model_name, samples[0],
                   np.array(mle).flatten()[0], np.array(mle_std).flatten()[0],
                   np.array(original_parameter).flatten()[0], np.array(crlb).flatten()[0])
    fig.tight_layout()
    plt.savefig(figure_output_pjoin('{}_{}_{}.png'.format(dataset_name, model_name, snr)), dpi=80)
    # plt.show()


model_names = [
    # 'BallStick_r1',
    # 'BallStick_r2',
    # 'BallStick_r3',
    # 'NODDI',
    'BinghamNODDI_r1',
    # 'Tensor',
    # 'CHARMED_r1',
    # 'CHARMED_r2'
]

map_titles = {
    'BallStick_r1': 'FS',
    'BallStick_r2': 'FS',
    'BallStick_r3': 'FS',
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

protocols = ['hcp_mgh_1003', 'rheinland_v3a_1_2mm']
# snrs = [2, 5, 10, 20, 30, 40, 50]
snrs = [30]
#
for model_name in model_names:
    for protocol in protocols:
        for snr in snrs:
            make_figure(protocol, model_name, snr)

        plot_names = ['{}_{}_{}.png'.format(protocol, model_name, snr) for snr in snrs]
        # subprocess.Popen('''
        #     convert +append {inputs} {protocol}_{model_name}.png
        # '''.format(inputs=' '.join(plot_names), protocol=protocol, model_name=model_name),
        #                  shell=True, cwd=figure_output_pjoin()).wait()

# for protocol in protocols:
#     plot_names = ['{}_{}.png'.format(protocol, model_name) for model_name in model_names]
#     subprocess.Popen('''
#         convert -append {inputs} {protocol}.png
#     '''.format(inputs=' '.join(plot_names), protocol=protocol),
#              shell=True, cwd=figure_output_pjoin()).wait()