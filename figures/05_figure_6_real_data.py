import glob
import mdt
import os
from mdt.lib.batch_utils import BatchFitProtocolLoader, SimpleSubjectInfo
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()
from mdt.lib.batch_utils import SimpleBatchProfile
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

figure_output_pjoin = mdt.make_path_joiner('/tmp/uncertainty_paper/std_snr/', make_dirs=True)


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)


class RheinLandBatchProfile(SimpleBatchProfile):

    def __init__(self, *args, resolutions_to_use=None, **kwargs):
        """Construct the Rheinland study batch profile.

        Args:
            resolutions_to_use (list of str): the list of resolutions to use, should contain
                'data_ms15' and/or 'data_ms20'. If not set, we will use both resolutions.
        """
        super(RheinLandBatchProfile, self).__init__(*args, **kwargs)
        self._auto_append_mask_name_to_output_sub_dir = False
        self._resolutions_to_use = resolutions_to_use or ['data_ms15', 'data_ms20']

    def _get_subjects(self, data_folder):
        dirs = sorted([os.path.basename(f) for f in glob.glob(os.path.join(data_folder, '*'))])
        subjects = []

        for directory in dirs:
            for resolution in self._resolutions_to_use:
                subject_pjoin = mdt.make_path_joiner(data_folder, directory, resolution)

                if os.path.exists(subject_pjoin()):
                    niftis = glob.glob(subject_pjoin('*.nii*'))

                    dwi_fname = list(filter(lambda v: '_mask' not in v and 'grad_dev' not in v, niftis))[0]
                    mask_fname = list(sorted(filter(lambda v: '_mask' in v, niftis)))[0]

                    protocol_fname = glob.glob(subject_pjoin('*prtcl'))[0]
                    protocol_loader = BatchFitProtocolLoader(subject_pjoin(), protocol_fname=protocol_fname)

                    subjects.append(SimpleSubjectInfo(data_folder, subject_pjoin(),
                                                      directory + '_' + resolution,
                                                      dwi_fname, protocol_loader, mask_fname))

        return subjects

    def __str__(self):
        return 'Rheinland'


def estimate_snr_wm_mask(input_data, wm_mask):
    unweighted_ind = input_data.protocol.get_unweighted_indices()
    unweighted_volumes = input_data.signal4d[..., unweighted_ind]

    snr = np.mean(unweighted_volumes, axis=-1) / np.std(unweighted_volumes, axis=-1)
    return mdt.create_roi(snr, wm_mask)


noise_snrs = [5, 10, 20, 30, 40, 50]
# noise_snrs = np.arange(5, 50, 5)


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
]
#
# protocol_names = {'hcp_mgh_1003': 'HCP MGH',
#                   'rheinland_v3a_1_2mm': 'RLS-pilot'}

model_titles = {
    'BallStick_r1': 'BallStick_in1',
    'BallStick_r2': 'BallStick_in2',
    'BallStick_r3': 'BallStick_in3',
    'Tensor': 'Tensor',
    'NODDI': 'NODDI',
    'BinghamNODDI_r1': 'Bingham-NODDI',
    'CHARMED_r1': 'CHARMED_in1',
}
#
# y_lims = {
#     'BallStick_r1': 0.061,
#     'BallStick_r2': 0.09,
#     'BallStick_r3': 0.11,
#     'Tensor': 0.32,
#     'NODDI': 0.5,
#     'BinghamNODDI_r1': 0.07,
#     'CHARMED_r1': 0.07
# }

def get_mcmc_std(results_pjoin, model_name):
    model_defined_maps = mdt.load_volume_maps(results_pjoin(model_name, 'samples', 'model_defined_maps'))
    univariate_normal_maps = mdt.load_volume_maps(results_pjoin(model_name, 'samples', 'univariate_normal'))

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
    elif model_name == 'BinghamNODDI_r1':
        return univariate_normal_maps['w_in0.w.std']


def get_mle_std(results_pjoin, model_name):
    maps = mdt.load_volume_maps(results_pjoin(model_name))

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
    elif model_name == 'BinghamNODDI_r1':
        return maps['w_in0.w.std']


def single_subject_results(subject_info):
    output_folder = subject_info.data_folder[:-1] + '_output'
    subject_id = subject_info.subject_id

    input_data = subject_info.get_input_data()
    wm_mask = output_folder + '/' + subject_id + '/wm_mask'
    snrs = estimate_snr_wm_mask(input_data, wm_mask)
    results_pjoin = mdt.make_path_joiner(output_folder + '/' + subject_id)

    results = {}

    for model in model_names:
        results_per_method = {}
        for method in ['sample', 'optimization']:
            results_per_snr = {}

            if method == 'sample':
                std_data = get_mcmc_std(results_pjoin, model)
            else:
                std_data = get_mle_std(results_pjoin, model)

            std_data = mdt.create_roi(std_data, wm_mask)

            for snr in noise_snrs:
                bin_width = 2.5
                std_values_in_snr = std_data[np.where((snr - bin_width <= snrs) & (snrs <= snr + bin_width))[0]]

                cutoff = 1
                std_values_in_snr = std_values_in_snr[
                    np.isfinite(std_values_in_snr) & (std_values_in_snr < cutoff) & (std_values_in_snr > 0)]

                results_per_snr.update({snr: np.mean(std_values_in_snr)})

            results_per_method[method] = results_per_snr
        results[model] = results_per_method
    return results


rls_results = mdt.batch_apply('/home/robbert/phd-data/rheinland/', single_subject_results,
                              batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
                              subjects_selection=range(10))

hcp_results = mdt.batch_apply('/home/robbert/phd-data/hcp_mgh/', single_subject_results,
                              batch_profile='HCP_MGH',
                              subjects_selection=range(10))


def create_figures(rls_results, hcp_results):
    set_matplotlib_font_size(22)
    colors = ['#c45054', '#0773b3', '#e79f27', '#65e065']
    linestyles = ['-', '-', '--', '--']

    for model in model_names:
        fig, axis = plt.subplots(1, 1)
        fig.set_size_inches(6, 5.5, forward=True)

        axis.set_title('{}'.format(model_titles[model]))
        axis.set_xlabel('SNR (a.u.)')
        axis.set_ylabel('Average std. (a.u.)')

        model_specific_protocols = [hcp_results, rls_results]
        if model.startswith('CHARMED'):
            model_specific_protocols = [hcp_results]

        plot_ind = 0
        for protocol_ind, results in enumerate(model_specific_protocols):
            for method in ['optimization', 'sample']:

                all_means = []
                for subject_id in results:
                    all_means.append([results[subject_id][model][method][snr] for snr in noise_snrs])

                x_axis = np.array(range(len(noise_snrs)))
                means = np.mean(all_means, axis=0)
                sems = np.array(np.std(all_means, axis=0) / np.sqrt(len(all_means)))

                if protocol_ind == 1:
                    x_axis = x_axis[1:]
                    means = means[1:]
                    sems = sems[1:]

                _, caps, _ = axis.errorbar(x_axis,
                                           means, marker='*', markersize=12,
                                           capthick=3,
                                           elinewidth=2, capsize=5,
                                           linewidth=2,
                                           yerr=sems,
                                           color=colors[plot_ind],
                                           linestyle=linestyles[plot_ind],
                                           # uplims=True,
                                           label='{} - {}'.format(str(protocol_ind), method.capitalize())
                                           )

                # caps[0].set_marker('_')
                # caps[0].set_markersize(20)

                # axis.set_xlim(-0.25, len(noise_snrs) +1 - 1 + .25)
                axis.set_xlim(-0.25, len(noise_snrs) - 1 + .25)

                def format_fn(tick_val, tick_pos):
                    # labels = [5] + noise_snrs
                    labels = noise_snrs
                    if int(tick_val) in range(len(noise_snrs)):
                        return labels[int(tick_val)]
                    else:
                        return ''

                axis.xaxis.set_major_formatter(FuncFormatter(format_fn))
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                plot_ind += 1

            # axis.legend()

        # axis.set_ylim(axis.get_ylim()[0], y_lims[model])
        plt.gcf().subplots_adjust(bottom=0.15, top=0.9, left=0.18, right=0.94)
        plt.savefig(figure_output_pjoin('{}.png'.format(model)), dpi=80)
        plt.close()
        # plt.show()

create_figures(rls_results, hcp_results)

