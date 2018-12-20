import subprocess

from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator
import pylab
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


input_pjoin = mdt.make_path_joiner('/home/robbert/programming/python/uncertainty_paper/data/snr_simulations/')
output_base_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/')
figure_output_pjoin = mdt.make_path_joiner('/tmp/uncertainty_paper/snr_simulations/', make_dirs=True)

nmr_trials = 2
simulations_unweighted_signal_height = 1e4

# noise_snrs = [2, 5, 10, 20, 30, 40, 50]
noise_snrs = [5, 10, 20, 30, 40, 50]


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
    'BinghamNODDI_r1': 'Bingham NODDI',
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


def get_mean_std(protocol_name, model_name, snr, trial_ind):
    model_defined_maps = mdt.load_volume_maps(output_base_pjoin(
        protocol_name, model_name, 'output', str(snr), str(trial_ind), model_name, 'samples', 'model_defined_maps'))
    univariate_normal_maps = mdt.load_volume_maps(output_base_pjoin(
        protocol_name, model_name, 'output', str(snr), str(trial_ind), model_name, 'samples', 'univariate_normal'))

    if model_name == 'Tensor':
        return model_defined_maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return model_defined_maps['FS.std']
    elif model_name == 'BallStick_r2':
        return model_defined_maps['FS.std']
    elif model_name == 'BallStick_r3':
        return model_defined_maps['FS.std']
    elif model_name == 'NODDI':
        return univariate_normal_maps['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return model_defined_maps['FR.std']
    elif model_name == 'CHARMED_r2':
        return model_defined_maps['FR.std']
    elif model_name == 'CHARMED_r3':
        return model_defined_maps['FR.std']
    elif model_name == 'BinghamNODDI_r1':
        return univariate_normal_maps['w_in0.w.std']


def get_mode_std(protocol_name, model_name, snr, trial_ind):
    maps = mdt.load_volume_maps(output_base_pjoin(protocol_name, model_name, 'output',
                                                  str(snr), str(trial_ind), model_name))

    if model_name == 'Tensor':
        return maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return maps['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        return maps['FS.std']
    elif model_name == 'BallStick_r3':
        return maps['FS.std']
    elif model_name == 'NODDI':
        return maps['w_ic.w.std']
    elif model_name == 'CHARMED_r1':
        return maps['FR.std']
    elif model_name == 'CHARMED_r2':
        return maps['FR.std']
    elif model_name == 'CHARMED_r3':
        return maps['FR.std']
    elif model_name == 'BinghamNODDI_r1':
        return maps['w_in0.w.std']


def get_results():
    results = {}

    for model in model_names:
        results_per_protocol = {}

        model_specific_protocols = protocols
        if model.startswith('CHARMED'):
            model_specific_protocols = ['hcp_mgh_1003']

        for protocol in model_specific_protocols:
            results_per_method = {}

            for method in ['sample', 'optimization']:
                results_per_snr = {}

                for snr in noise_snrs:

                    trial_mean_stds = []
                    trial_std_stds = []
                    for trial_ind in range(nmr_trials):
                        if method == 'sample':
                            std_data = get_mean_std(protocol, model, snr, trial_ind)
                        else:
                            std_data = get_mode_std(protocol, model, snr, trial_ind)

                        std_data_regulated = std_data[np.isfinite(std_data) & (std_data < 1)]
                        trial_mean_stds.append(np.mean(std_data_regulated))
                        trial_std_stds.append(np.std(std_data_regulated))

                    results_per_snr.update({snr: {'mean': np.mean(trial_mean_stds),
                                                  'sem': np.mean(trial_std_stds) / np.sqrt(nmr_trials)}})

                results_per_method[method] = results_per_snr
            results_per_protocol[protocol] = results_per_method
        results[model] = results_per_protocol
    return results


def create_figures(results):
    # fig, axarr = plt.subplots(2, 2)
    # fig.subplots_adjust(wspace=0.25, left=0.06, right=0.98, hspace=0.4, bottom=0.08)
    # fig.suptitle('Effect of SNR on std.', y=0.96)
    # ax_ind = list(np.ndindex(*axarr.shape))
    set_matplotlib_font_size(22)
    colors = ['#c45054', '#0773b3', '#e79f27', '#65e065']
    linestyles = ['-', '-', '--', '--']

    for model in model_names:
        fig, axis = plt.subplots(1, 1)
        fig.set_size_inches(6, 5.5, forward=True)

        axis.set_title('{}'.format(model_titles[model]))
        # axis.set_xlabel('SNR (a.u.)')
        # axis.set_ylabel('Average std. (a.u.)')

        model_specific_protocols = protocols
        if model.startswith('CHARMED'):
            model_specific_protocols = ['hcp_mgh_1003']

        plot_ind = 0
        for protocol_ind, protocol in enumerate(model_specific_protocols):
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
                                           # uplims=True,
                                           label='{} - {}'.format(protocol_names[protocol], method.capitalize()))

                # caps[0].set_marker('_')
                # caps[0].set_markersize(20)

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

            # axis.legend()
        plt.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.18, right=0.94)
        plt.savefig(figure_output_pjoin('{}.png'.format(model)), dpi=80)
        plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_ind = 0
    for protocol_ind, protocol in enumerate(protocols):
        for method in ['optimization', 'sample']:
            ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10),
                    color=colors[plot_ind],
                    linestyle=linestyles[plot_ind],
                    label='{} - {}'.format(protocol_names[protocol], method.capitalize()))
            plot_ind += 1

    figlegend = plt.figure(figsize=(5, 5))
    figlegend.legend(*ax.get_legend_handles_labels(), loc='center', fontsize='small')
    figlegend.savefig(figure_output_pjoin('legend.png'))

    plt.close()

results = get_results()
create_figures(results)
