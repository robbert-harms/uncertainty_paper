import os
import multiprocessing as mp
from itertools import product, repeat
from textwrap import dedent

import numpy as np
from matplotlib.ticker import MaxNLocator, LinearLocator
from scipy.stats import ttest_rel, ttest_ind
from sklearn.neighbors import KernelDensity
import mdt
import matplotlib.pyplot as plt
import seaborn

from mot.stats import gaussian_overlapping_coefficient

seaborn.set()


__author__ = 'Robbert Harms'
__date__ = '2018-11-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)


set_matplotlib_font_size(22)


datasets = [
    # 'hcp/mgh_1003',
    'rls/v3a_1_data_ms20'
]

dataset_names = {
    'hcp/mgh_1003': 'hcp',
    'rls/v3a_1_data_ms20': 'rls',
}

masks = {
    'hcp/mgh_1003': '/home/robbert/phd-data/hcp_mgh/mgh_1003/diff/preproc/mri/diff_preproc_mask.nii',
    'rls/v3a_1_data_ms20': '/home/robbert/phd-data/rheinland/v3a_1/data_ms20/DICOM_CMRR_3shell_20iso_tra_mb3_20150921095022_8_eddy_corrected_sortb0_brain_mask.nii.gz',
}

model_names = [
    # 'BallStick_r1',
    # 'BallStick_r2',
    # 'BallStick_r3',
    # 'Tensor',
    # 'NODDI',
    # 'BinghamNODDI_r1',
    'CHARMED_r1'
]

model_titles = {
    'BallStick_r1': 'BallStick_in1',
    'BallStick_r2': 'BallStick_in2',
    'BallStick_r3': 'BallStick_in3',
    'Tensor': 'Tensor',
    'NODDI': 'NODDI',
    'BinghamNODDI_r1': 'BinghamNODDI',
    'CHARMED_r1': 'CHARMED_in1',
    'CHARMED_r2': 'CHARMED_in2',
    'CHARMED_r3': 'CHARMED_in3'
}

model_maps = {
    'BallStick_r1': {
        'FS': 'FS',
        'w_stick0.w': 'Stick0.w'
    },
    'BallStick_r2': {
        'FS': 'FS',
        'w_stick0.w': 'Stick0.w',
        'w_stick1.w': 'Stick1.w'
    },
    'BallStick_r3': {
        'FS': 'FS',
        'w_stick0.w': 'Stick0.w',
        'w_stick1.w': 'Stick1.w',
        'w_stick2.w': 'Stick2.w'
    },
    'Tensor': {
        # 'Tensor.AD': 'AD',
        # 'Tensor.d': 'Tensor.d',
        # 'Tensor.dperp0': 'Tensor.dperp0',
        # 'Tensor.dperp1': 'Tensor.dperp2',
        'Tensor.theta': 'Tensor.theta',
        'Tensor.phi': 'Tensor.phi',
        'Tensor.psi': 'Tensor.psi',
        # 'Tensor.FA': 'FA',
        # 'Tensor.MD': 'MD',
        # 'Tensor.RD': 'RD'
    },
    'NODDI': {
        # 'NDI': 'NDI',
        # 'ODI': 'ODI',
        'w_ec.w': 'w_ec.w',
        'w_ic.w': 'w_ic.w',
        # 'w_csf.w': 'w_csf.w',
    },
    'BinghamNODDI_r1': {
        # 'w_en0.w': 'w_en0.w',
        'w_in0.w': 'FR',
    },
    'CHARMED_r1': {
        'FR': 'FR'
    }
}

parameter_boundaries = {
    'BallStick_r1': {'FS': [0, 1],
                     'w_stick0.w': [0, 1]},
    'BallStick_r2': {'FS': [0, 1],
                     'w_stick0.w': [0, 1],
                     'w_stick1.w': [0, 1]},
    'BallStick_r3': {'FS': [0, 1],
                     'w_stick0.w': [0, 1],
                     'w_stick1.w': [0, 1],
                     'w_stick2.w': [0, 1]},
    'Tensor': {'Tensor.AD': [0, 1e-8],
               'Tensor.d': [1e-12, 1e-8],
               'Tensor.dperp0': [0, 1e-8],
               'Tensor.dperp1': [0, 1e-8],
               'Tensor.theta': [0, np.pi],
               'Tensor.phi': [0, np.pi],
               'Tensor.psi': [0, np.pi],
               'Tensor.FA': [0, 1],
               'Tensor.MD': [0, 1e-8],
               'Tensor.RD': [0, 1e-8],},
    'NODDI': {'NDI': [0, 1],
              'ODI': [0, 1],
              'w_ec.w': [0, 1],
              'w_ic.w': [0, 1],
              'w_csf.w': [0, 1]},
    'BinghamNODDI_r1': {'w_in0.w': [0, 1]},
    'CHARMED_r1': {'FR': [0, 1]}
}


kernel_density_bandwidth = {
    'Tensor': {'Tensor.AD': 0.001 * 0.5e-8,
               'Tensor.d': 0.001 * 0.5e-8,
               'Tensor.dperp0': 0.001 * 0.5e-8,
               'Tensor.dperp1': 0.001 * 0.5e-8,
               'Tensor.MD': 0.001 * 0.5e-8,
               'Tensor.RD': 0.001 * 0.5e-8}
}


plot_limits = {
    'BallStick_r1': {
        'FS': {'': (0, 1), 'std': (0, 0.06)},
        'w_stick0.w': {'': (0, 1), 'std': (0, 0.06)},
    },
    'BallStick_r2': {
        'FS': {'': (0, 1), 'std': (0, 0.06)},
        'w_stick0.w': {'': (0, 1), 'std': (0, 0.06)},
        'w_stick1.w': {'': (0, 1), 'std': (0, 0.06)},
    },
    'BallStick_r3': {
        'FS': {'': (0, 1), 'std': (0, 0.06)},
        'w_stick0.w': {'': (0, 1), 'std': (0, 0.06)},
        'w_stick1.w': {'': (0, 1), 'std': (0, 0.06)},
        'w_stick2.w': {'': (0, 1), 'std': (0, 0.06)},
    },
    'Tensor': {
        'Tensor.AD': {'': (1e-10, 0.5e-8), 'std': (0, 0.5e-9)},
        'Tensor.d': {'': (1e-10, 0.5e-8), 'std': (0, 0.5e-9)},
        'Tensor.dperp0': {'': (1e-10, 0.5e-8), 'std': (0, 0.5e-9)},
        'Tensor.dperp1': {'': (1e-10, 0.5e-8), 'std': (0, 0.5e-9)},
        'Tensor.theta': {'': (0, np.pi), 'std': (0, np.pi)},
        'Tensor.phi': {'': (0, np.pi), 'std': (0, np.pi)},
        'Tensor.psi': {'': (0, np.pi), 'std': (0, np.pi)},
        'Tensor.FA': {'': (0, 1), 'std': (0, 0.15)},
        'Tensor.MD': {'': (1e-10, 0.5e-8), 'std': (0, 1e-4)},
        'Tensor.RD': {'': (1e-10, 0.5e-8), 'std': (0, 0.5e-9)}
    },
    'NODDI': {
        'NDI': {'': (0, 1), 'std': (0, 0.12)},
        'ODI': {'': (0, 1), 'std': (0, 0.12)},
        'w_ec.w': {'': (0, 1), 'std': (0, 0.12)},
        'w_ic.w': {'': (0, 1), 'std': (0, 0.12)},
        'w_csf.w': {'': (0, 1), 'std': (0, 0.12)},
    },
    'BinghamNODDI_r1': {
        'w_in0.w': {'': (0, 1), 'std': (0, 0.15)},
    },
    'CHARMED_r1': {
        'FR': {'': (0, 1), 'std': (0, 0.15)},
    }
}

output_base_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/single_subject/')
figure_output_pjoin = mdt.make_path_joiner('/tmp/uncertainty_paper/mle_mcmc_scatter/', make_dirs=True)


def get_mle_results(dataset_name, model_name, map_name):
    maps = mdt.load_volume_maps(output_base_pjoin(dataset_name, model_name))
    return maps[map_name], maps[map_name + '.std']


def get_mcmc_results(dataset_name, model_name, map_name):
    model_defined_maps = mdt.load_volume_maps(output_base_pjoin(
        dataset_name, model_name, 'samples', 'model_defined_maps'))
    univariate_normal_maps = mdt.load_volume_maps(output_base_pjoin(
        dataset_name, model_name, 'samples', 'univariate_normal'))

    if map_name in model_defined_maps.keys():
        return model_defined_maps[map_name], model_defined_maps[map_name + '.std']
    else:
        return univariate_normal_maps[map_name], univariate_normal_maps[map_name + '.std']


def kde_sklearn(x, bandwidth=0.1, **kwargs):
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x)
    log_pdf = kde_skl.score_samples(x)
    return np.exp(log_pdf)


def scatter(x, y, dataset, model_name, map_name, data_type, color_coding=True):
    title = '{} - {} ({})'.format(
        dataset_names[dataset],
        model_titles[model_name],
        model_maps[model_name][map_name])

    if data_type == 'std':
        title += ' std.'

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 6.5, forward=True)

    x = np.squeeze(x)
    y = np.squeeze(y)

    limits = plot_limits[model_name][map_name][data_type]
    selection = ((x >= limits[0]) & (x <= limits[1]) & (y >= limits[0]) & (y <= limits[1]))
    x = x[selection]
    y = y[selection]

    if color_coding:
        ## plot with local density color coding
        bandwidth = kernel_density_bandwidth.get(model_name, {}).get(map_name, 0.001)
        z = kde_sklearn(np.hstack([x[:, None], y[:, None]]), bandwidth=bandwidth)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, edgecolor=[], cmap='Spectral_r')
    else:
        ## plot without color coding (faster)
        ax.scatter(x, y, edgecolor=[], cmap='Spectral_r')

    ax.set_xlabel('MLE')
    ax.set_ylabel('MCMC')

    ax.set_xlim(*limits)
    ax.set_ylim(*limits)

    if data_type == 'std':
        if 'BinghamNODDI_r1' in model_name and 'w_in0.w' in map_name:
            ax.yaxis.set_major_locator(LinearLocator(4))
        elif 'NODDI' in model_name:
            ax.yaxis.set_major_locator(MaxNLocator(5))
        elif 'CHARMED' in model_name:
            ax.yaxis.set_major_locator(LinearLocator(4))
        elif model_name == 'Tensor':
            ax.yaxis.set_major_locator(LinearLocator(4))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(4))

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-')
    ax.set_title(title)

    pcc = pearson_correlation_coefficient(x, y)
    ax.text(0.05, 0.9, 'ρ = {:.3f}'.format(pcc), transform=ax.transAxes)

    ax.get_yticklabels()[-1].set_verticalalignment('top')
    ax.get_yticklabels()[0].set_verticalalignment('center')
    ax.get_xticklabels()[0].set_horizontalalignment('center')

    plt.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.187, right=0.94)
    # plt.savefig(figure_output_pjoin('{}.png'.format(title)), dpi=80)
    # plt.close()
    plt.show()


def pearson_correlation_coefficient(x, y):
    coef_mat = np.corrcoef(np.squeeze(x), np.squeeze(y))
    return coef_mat[0, 1]


def process_model(dataset, model_name, map_name):
    mle, mle_std = get_mle_results(dataset, model_name, map_name)
    mcmc, mcmc_std = get_mcmc_results(dataset, model_name, map_name)

    wm_mask = mdt.load_brain_mask(output_base_pjoin(dataset, 'wm_mask'))

    no_filter_count = np.count_nonzero(wm_mask)

    low_bound, high_bound = parameter_boundaries[model_name][map_name]
    offset = 0.01 * high_bound

    wm_mask *= (mle >= (low_bound + offset))[..., 0]
    wm_mask *= (mle <= (high_bound - offset))[..., 0]
    wm_mask *= (mcmc >= (low_bound + offset))[..., 0]
    wm_mask *= (mcmc <= (high_bound - offset))[..., 0]
    border_filter_count = np.count_nonzero(wm_mask)

    unlikely_filter_count = np.count_nonzero(wm_mask)

    mdt.view_maps({'mle': mle, 'mle_std': mle_std, 'mcmc': mcmc, 'mcmc_std': mcmc_std})

    items = mdt.create_roi({'mle': mle, 'mle_std': mle_std, 'mcmc': mcmc, 'mcmc_std': mcmc_std}, wm_mask)

    diff = items['mle_std'] - items['mcmc_std']
    diff[np.abs(diff) > 1] = 0

    nmr_outliers = np.sum(np.abs(diff - np.mean(diff)) >= (2 * np.std(diff)))
    percentage_outliers = np.round(100 - 100 * nmr_outliers / len(items['mle_std']), 1)

    print(dedent(f'''
        Processing {dataset} {model_name}, {map_name}
        White matter mask count: {no_filter_count}
        Border filter count, {no_filter_count - border_filter_count}
        Unlikely count, {border_filter_count - unlikely_filter_count}
        Nmr outliers, {nmr_outliers}
        Nmr voxels, {len(items['mle_std'])}
        Percentage_outliers, {percentage_outliers}
    '''))

    scatter(items['mle'], items['mcmc'], dataset, model_name, map_name, '', color_coding=False)
    scatter(items['mle_std'], items['mcmc_std'], dataset, model_name, map_name, 'std', color_coding=False)


items = []
for dataset in datasets:
    for model_name in model_names:
        for map_name in model_maps[model_name].keys():
            items.append((dataset, model_name, map_name))

for item in items:
    process_model(*item)

# pool = mp.Pool(processes=18)
# pool.starmap(process_model, items)
# pool.close()
# pool.join()
