import os

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
    'hcp/mgh_1003',
    'rls/v3a_1_data_ms20'
]

dataset_names = {
    'hcp/mgh_1003': 'hcp',
    'rls/v3a_1_data_ms20': 'rls',
}

masks = {
    'hcp/mgh_1003': '/home/robbert/phd-data/hcp_mgh/mgh_1003/diff/preproc/mri/diff_preproc_mask.nii',
    'rls/v3a_1_data_ms20': '/home/robbert/phd-data/rheinland/v3a_1/data_ms20/DICOM_CMRR_3shell_20iso_tra_mb3_20150921095022_8_eddy_corrected_sortb0_brain_mask.nii.gz',

    # 'hcp/mgh_1003': '/home/robbert/phd-data/hcp_mgh/mgh_1003/diff/preproc/diff_preproc_mask_2_44.nii.gz',
    # 'rls/v3a_1_data_ms20': '/home/robbert/phd-data/rheinland/v3a_1/data_ms20/DICOM_CMRR_3shell_20iso_tra_mb3_20150921095022_8_eddy_corrected_sortb0_brain_mask_44_2.nii.gz',
}

model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'Tensor',
    'NODDI',
    'BinghamNODDI_r1',
    'CHARMED_r1'
]

map_names = {
    'BallStick_r1': 'FS',
    'BallStick_r2': 'FS',
    'BallStick_r3': 'FS',
    'Tensor': 'FA',
    'NODDI': 'FR',
    'BinghamNODDI_r1': 'FR',
    'CHARMED_r1': 'FR',
    'CHARMED_r2': 'FR',
    'CHARMED_r3': 'FR'
}


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


output_base_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/single_subject/')
# output_base_pjoin = mdt.make_path_joiner('/home/robbert/Downloads/single_subject/')

figure_output_pjoin = mdt.make_path_joiner('/tmp/uncertainty_paper/mle_mcmc_scatter/', make_dirs=True)
goc_output_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/single_subject_gocs/')


def get_mle_results(dataset_name, model_name):
    maps = mdt.load_volume_maps(output_base_pjoin(dataset_name, model_name))

    if model_name == 'Tensor':
        return maps['Tensor.FA'], maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return maps['w_stick0.w'], maps['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        return maps['FS'], maps['FS.std']
        # return maps['w_stick0.w'], maps['w_stick0.w.std']
    elif model_name == 'BallStick_r3':
        return maps['FS'], maps['FS.std']
        # return maps['w_stick0.w'], maps['w_stick0.w.std']
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
        # return univariate_normal_maps['w_stick0.w'], univariate_normal_maps['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        return model_defined_maps['FS'], model_defined_maps['FS.std']
        # return univariate_normal_maps['w_stick0.w'], univariate_normal_maps['w_stick0.w.std']
    elif model_name == 'BallStick_r3':
        return model_defined_maps['FS'], model_defined_maps['FS.std']
        # return univariate_normal_maps['w_stick0.w'], univariate_normal_maps['w_stick0.w.std']
    elif model_name == 'NODDI':
        return univariate_normal_maps['w_ic.w'], univariate_normal_maps['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return model_defined_maps['FR'], model_defined_maps['FR.std']
    elif model_name == 'CHARMED_r2':
        return model_defined_maps['FR'], model_defined_maps['FR.std']
    elif model_name == 'BinghamNODDI_r1':
        return univariate_normal_maps['w_in0.w'], univariate_normal_maps['w_in0.w.std']


plot_configs = {
    'BallStick_r1': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.04), 'y': (0, 0.04)},
    },
    'BallStick_r2': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.04), 'y': (0, 0.04)},
    },
    'BallStick_r3': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.04), 'y': (0, 0.04)},
    },
    'Tensor': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.15), 'y': (0, 0.15)},
    },
    'NODDI': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.05), 'y': (0, 0.05)},
    },
    'BinghamNODDI_r1': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.05), 'y': (0, 0.05)},
    },
    'CHARMED_r1': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.1), 'y': (0, 0.1)},
    },
    'CHARMED_r2': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.1), 'y': (0, 0.1)},
    },
    'CHARMED_r3': {
        '': {'x': (0, 1), 'y': (0, 1)},
        'std': {'x': (0, 0.1), 'y': (0, 0.1)},
    }
}


def kde_sklearn(x, bandwidth=0.1, **kwargs):
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x)
    log_pdf = kde_skl.score_samples(x)
    return np.exp(log_pdf)


def scatter(x, y, dataset, model_name, data_type):
    title = '{} - {} ({})'.format(
        dataset_names[dataset],
        model_titles[model_name],
        map_names[model_name])

    if data_type == 'std':
        title += ' std.'

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 6.5, forward=True)

    x = np.squeeze(x)
    y = np.squeeze(y)

    plot_config = plot_configs[model_name][data_type]

    # local density estimates
    # z = kde_sklearn(np.hstack([x[:, None], y[:, None]]), bandwidth=0.001)


    # z_limit = 12.5
    # print('filtering {}%'.format(np.count_nonzero(z < z_limit) / len(z) * 100))
    # x = x[z > z_limit]
    # y = y[z > z_limit]
    # z = z[z > z_limit]
    # idx = z.argsort()[int(0.005 * len(z)):]

    # idx = z.argsort()

    # x, y, z = x[idx], y[idx], z[idx]
    # ax.scatter(x, y, c=z, edgecolor='', cmap='Spectral_r')

    ## plot without color coding (faster)
    ax.scatter(x, y, edgecolor='', cmap='Spectral_r')

    ax.set_xlabel('MLE')
    ax.set_ylabel('MCMC')

    ax.set_xlim(*plot_config['x'])
    ax.set_ylim(*plot_config['y'])

    if data_type == 'std':
        if 'NODDI' in model_name:
            ax.yaxis.set_major_locator(MaxNLocator(5))
        if 'CHARMED' in model_name:
                ax.yaxis.set_major_locator(MaxNLocator(5))
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

    ax.get_yticklabels()[-1].set_verticalalignment('top')
    ax.get_yticklabels()[0].set_verticalalignment('center')
    ax.get_xticklabels()[0].set_horizontalalignment('center')

    plt.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.187, right=0.94)
    plt.savefig(figure_output_pjoin('{}.png'.format(title)), dpi=80)
    plt.close()
    # plt.show()


def pearson_correlation_coefficient(x, y, dataset, model_name, data_type):
    coef_mat = np.corrcoef(np.squeeze(x), np.squeeze(y))
    print(dataset, model_name, data_type, coef_mat[0, 1])


percentages = []
for dataset in datasets:
    for model_name in model_names:
        print(dataset, model_name)

        mle, mle_std = get_mle_results(dataset, model_name)
        mcmc, mcmc_std = get_mcmc_results(dataset, model_name)

        wm_mask = mdt.load_brain_mask(output_base_pjoin(dataset, 'wm_mask'))
        # wm_mask = mdt.load_brain_mask(masks[dataset])
        # wm_mask *= (np.abs(mle - mcmc) < 0.01)[..., 0]
        # wm_mask *= (mle >= 0.1)[..., 0]
        # wm_mask *= (mle <= 0.9)[..., 0]
        # wm_mask *= (mcmc >= 0.1)[..., 0]
        # wm_mask *= (mcmc <= 0.9)[..., 0]

        # wm_mask = mle >= 0.1

        # wm_mask *= (np.abs(mle_std - mcmc_std) < 0.01)[..., 0]


        # items = {'mle': mle, 'mle_std': mle_std, 'mcmc': mcmc, 'mcmc_std': mcmc_std,
        #          'p_diff': np.abs(mle - mcmc), 'std_diff': np.abs(mle_std - mcmc_std),
        #          'std_threshold_map': np.abs(mle_std - mcmc_std) > 0.01
        #          }
        # mdt.view_maps(items)
        # exit(0)

        items = mdt.create_roi({'mle': mle, 'mle_std': mle_std, 'mcmc': mcmc, 'mcmc_std': mcmc_std}, wm_mask)
        #
        # mle_stats = []
        # mcmc_stats = []
        #
        # for i in range(30):
        #     ind_mle = np.random.rand(len(items['mle_std'])) < 0.01
        #     ind_mcmc = np.random.rand(len(items['mle_std'])) < 0.01
        #
        #     mle_stats.append(np.mean(items['mle_std'][ind_mle]))
        #     mcmc_stats.append(np.mean(items['mcmc_std'][ind_mcmc]))
        #
        # print(ttest_ind(mle_stats, mcmc_stats))


        diff = items['mle_std'] - items['mcmc_std']
        # lim = np.abs(diff) < 0.5
        diff[np.abs(diff) > 1] = 0

        # items['mle_std'] = items['mle_std'][lim]
        # items['mcmc_std'] = items['mcmc_std'][lim]
        # diff = diff[lim]

        # items['mle_std'] = items['mle_std'][np.abs(diff - np.mean(diff)) < (2 * np.std(diff))]
        # items['mcmc_std'] = items['mcmc_std'][np.abs(diff - np.mean(diff)) < (2 * np.std(diff))]

        # items['mask'] = np.abs(diff - np.mean(diff)) >= (2 * np.std(diff))

        # mdt.view_maps(mdt.restore_volumes(items, wm_mask))
        # exit(0)

        nmr_outliers = np.sum(np.abs(diff - np.mean(diff)) >= (2 * np.std(diff)))
        percentage_outliers = np.round(100 - 100 * nmr_outliers / len(items['mle_std']), 1)
        percentages.append(percentage_outliers)
        print(nmr_outliers, len(items['mle_std']), '{}%'.format(percentage_outliers))

        # scatter(items['mle'], items['mcmc'], dataset, model_name, '')
        # scatter(items['mle_std'], items['mcmc_std'], dataset, model_name, 'std')
        # exit(0)
        # pearson_correlation_coefficient(items['mle'], items['mcmc'], dataset, model_name, '')
        # pearson_correlation_coefficient(items['mle_std'], items['mcmc_std'], dataset, model_name, 'std')

        # plt.hist(items['mle'], bins=1000, normed=True)
        # plt.hist(items['mcmc'], bins=1000, normed=True)
        #
        # plt.hist(items['mle_std'], bins=1000)
        # plt.hist(items['mcmc_std'], bins=1000)

        # plt.hist(items['mle_std'] - items['mcmc_std'], bins=1000)
        #
        # plt.show()
        # exit(0)

        # plt.plot(items['mle_std'][:1000])
        # plt.plot(items['mcmc_std'][:1000])
        # plt.show()

        # kl_divergence = np.log(items['mcmc_std'] / items['mle_std']) + (items['mle_std']**2 + (items['mle'] - items['mcmc'])**2) / (2 * items['mcmc_std']**2) - 0.5
        # kl_divergence[~np.isfinite(kl_divergence)] = 0
        # print(np.mean(kl_divergence), np.std(kl_divergence))
        # plt.plot(kl_divergence)
        # plt.show()
        # exit()

        # statistics, p_value = ttest_ind(items['mle_std'], items['mcmc_std'])
        # print(statistics, p_value)
        #
        # p_values = []
        # for i in range(0, len(items['mle_std']), 1000):
        #     statistics, p_value = ttest_rel(items['mle_std'][:i], items['mcmc_std'][:i])
        #     p_values.append(p_value)
        #     # print(statistics, p_value)
        # plt.plot(p_values)
        # plt.show()
        # exit(0)
        # goc_out_file = goc_output_pjoin(dataset_names[dataset] + ' ' + model_name + ' ' +
        #                                 map_names[model_name] + '.nii.gz')
        # if not os.path.exists(goc_out_file):
        #     items = mdt.create_roi({'mle': mle, 'mle_std': mle_std, 'mcmc': mcmc, 'mcmc_std': mcmc_std}, masks[dataset])
        #     goc = gaussian_overlapping_coefficient(np.array(items['mle']),
        #                                            np.array(items['mle_std']),
        #                                            np.array(items['mcmc']),
        #                                            np.array(items['mcmc_std']))
        #     mdt.write_nifti(
        #         mdt.restore_volumes(goc, masks[dataset]),
        #         goc_out_file
        #     )

        # items.update({'goc': goc})
        # mdt.view_maps(mdt.restore_volumes(items, masks[dataset]))

print(np.mean(percentages))

# plt.show()