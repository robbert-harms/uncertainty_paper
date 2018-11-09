import subprocess

from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

import mdt
from mdt.lib.post_processing import DTIMeasures
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

__author__ = 'Robbert Harms'
__date__ = '2018-11-09'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/')
nmr_trials = 10
simulations_unweighted_signal_height = 1e4

noise_snrs = [2, 5, 10, 20, 30, 40, 50]

protocols = [
    'hcp_mgh_1003',
    'rheinland_v3a_1_2mm'
]

model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'Tensor',
    'NODDI',
    'BinghamNODDI_r1',
    'CHARMED_r1',
    'CHARMED_r2',
    'CHARMED_r3'
]

protocol_names = {'hcp_mgh_1003': 'HCP MGH',
                  'rheinland_v3a_1_2mm': 'RLS-pilot'}

model_titles = {
    'BallStick_r1': 'BallStick_in1',
    'BallStick_r2': 'BallStick_in2',
    'BallStick_r3': 'BallStick_in3',
    'Tensor': 'Tensor',
    'NODDI': 'NODDI',
    'CHARMED_r1': 'CHARMED_in1',
    'CHARMED_r2': 'CHARMED_in2',
    'CHARMED_r3': 'CHARMED_in3'
}


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)

set_matplotlib_font_size(18)


def get_results():
    results = {}

    for model in model_names:
        results_per_protocol = {}

        for protocol in protocols:
            results_per_method = {}

            for method in ['sample', 'optimization']:
                results_per_snr = {}

                for snr in noise_snrs:

                    trial_mean_stds = []
                    trial_std_stds = []
                    for trial_ind in range(nmr_trials):
                        if method == 'sample':
                            path = pjoin(protocol, model, 'output', str(snr), str(trial_ind), model,
                                         'samples', 'model_defined_maps', model_std_map_names[model] + '.nii.gz')
                        else:
                            path = pjoin(protocol, model, 'output', str(snr), str(trial_ind),
                                         model, model_std_map_names[model] + '.nii.gz')

                        std_data = mdt.load_nifti(path).get_data()
                        std_data_regulated = std_data[np.logical_and(np.isfinite(std_data), std_data < 1)]
                        trial_mean_stds.append(np.mean(std_data_regulated))
                        trial_std_stds.append(np.std(std_data_regulated))

                    results_per_snr.update({snr: {'mean': np.mean(trial_mean_stds),
                                                  'sem': np.mean(trial_std_stds) / np.sqrt(nmr_trials)}})
                    # results_per_snr.update({snr: {'mean': trial_mean_stds[0],
                    #                               'sem': trial_sems_stds[0]}})

                results_per_method[method] = results_per_snr
            results_per_protocol[protocol] = results_per_method
        results[model] = results_per_protocol
    return results


def figure_all_merged(results):
    set_matplotlib_font_size(28)

    fig, axarr = plt.subplots(2, 2)
    fig.subplots_adjust(wspace=0.25, left=0.06, right=0.98, hspace=0.4, bottom=0.08)
    fig.suptitle('Effect of SNR on std.', y=0.96)
    ax_ind = list(np.ndindex(*axarr.shape))

    colors = ['#c45054', '#0773b3', '#e79f27', '#65e065']
    linestyles = ['-', '--', '-', '--']

    for ind, model in enumerate(models):
        axis = axarr[ax_ind[ind]]
        axis.set_title('{}'.format(model_titles[model]))
        axis.set_xlabel('SNR (a.u.)')
        axis.set_ylabel('Average std. (a.u.)')

        plot_ind = 0
        for protocol_ind, protocol in enumerate(protocols):
            for method in ['optimization', 'sample']:
                means = [results[model][protocol][method][snr]['mean'] for snr in noise_snrs]
                sems = [results[model][protocol][method][snr]['sem'] for snr in noise_snrs]

                _, caps, _ = axis.errorbar(np.array(range(len(noise_snrs))),# - 0.05 + (protocol_ind * 0.1),
                                           means, marker='*', markersize=12,
                                           capthick=3,
                                           elinewidth=2, capsize=5,
                                           linewidth=2,
                                           yerr=np.array(sems),#yerr,
                                           color=colors[plot_ind],
                                           linestyle=linestyles[plot_ind],
                                           label='{} - {}'.format(protocol_names[protocol], method.capitalize()))

                axis.set_xlim(-0.25, len(noise_snrs) - 1 + .25)

                def format_fn(tick_val, tick_pos):
                    labels = noise_snrs
                    if int(tick_val) in range(len(noise_snrs)):
                        return labels[int(tick_val)]
                    else:
                        return ''

                axis.xaxis.set_major_formatter(FuncFormatter(format_fn))
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                plot_ind += 1

            axis.legend()
    plt.show()


def figure_protocols_next_to_each_other(results):
    set_matplotlib_font_size(28)

    fig, axarr = plt.subplots(4, 2, sharex='all', sharey='row', figsize=(18, 8), dpi=80)
    fig.subplots_adjust(wspace=0.05, left=0.08, right=0.98, hspace=0.4, bottom=0.08)
    fig.suptitle('Effect of SNR on std.', y=0.96)
    # ax_ind = list(np.ndindex(*axarr.shape))
    #
    colors = ['#c45054', '#0773b3']#, '#e79f27', '#65e065']
    linestyles = ['-', '--']#, '-', '--']

    for model_ind, model in enumerate(models):
        for protocol_ind, protocol in enumerate(protocols):
            axis = axarr[(model_ind, protocol_ind)]
            axis.set_title('{}'.format(model_titles[model]))

            # if protocol_ind == 0:
            #     axis.set_ylabel(model_titles[model])

            if model_ind == 3:
                axis.set_xlabel('SNR (a.u.)')

            for method_ind, method in enumerate(['optimization', 'sample']):
                means = [results[model][protocol][method][snr]['mean'] for snr in noise_snrs]
                sems = [results[model][protocol][method][snr]['sem'] for snr in noise_snrs]

                _, caps, _ = axis.errorbar(np.array(range(len(noise_snrs))),  # - 0.05 + (protocol_ind * 0.1),
                                           means, marker='*', markersize=12,
                                           capthick=3,
                                           elinewidth=2, capsize=5,
                                           linewidth=2,
                                           yerr=np.array(sems),  # yerr,
                                           color=colors[method_ind],
                                           linestyle=linestyles[method_ind],
                                           label=method.capitalize())

                def format_fn(tick_val, tick_pos):
                    labels = noise_snrs
                    if int(tick_val) in range(len(noise_snrs)):
                        return labels[int(tick_val)]
                    else:
                        return ''

                axis.set_xlim(-0.25, len(noise_snrs) - 1 + .25)
                axis.xaxis.set_major_formatter(FuncFormatter(format_fn))
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))

                axis.yaxis.set_major_locator(MaxNLocator(3))
                axis.get_yticklabels()[-1].set_verticalalignment('top')
                axis.get_yticklabels()[0].set_verticalalignment('bottom')

            if protocol_ind == 1 and model_ind == 0:
                axis.legend()

    fig.text(0.05, 0.5, 'Average std. (a.u.)', ha='center', va='center', rotation='vertical')

    plt.show()

results = get_results()
figure_all_merged(results)
# figure_protocols_next_to_each_other(results)