import matplotlib.pyplot as plt
import numpy as np

import mdt
from mt_scripts.robbert.papers.uncertainty_paper.configuration import mgh_data_pjoin, mgh_output_pjoin

subject_id = 'mgh_1003'
noise_std = 44.19256591796875

mask = mdt.load_brain_mask(mgh_data_pjoin(subject_id, 'diff', 'preproc', 'diff_preproc_mask'))
wm_mask = mdt.load_brain_mask(mgh_data_pjoin(subject_id, 'diff', 'preproc', 'output', 'optimization_paper', 'wm_mask'))

output_pjoin = mgh_output_pjoin.create_extended('mgh_1003')

input_data = mdt.load_input_data(mgh_data_pjoin(subject_id, 'diff', 'preproc', 'mri', 'diff_preproc.nii'),
                                 mdt.auto_load_protocol(mgh_data_pjoin(subject_id, 'diff', 'preproc')),
                                 mask,
                                 noise_std=noise_std)


unweighted_volumes = input_data.protocol.get_unweighted_indices()

snr = np.mean(input_data.signal4d[..., unweighted_volumes], axis=-1) \
      / np.std(input_data.signal4d[..., unweighted_volumes], axis=-1)
snr = mdt.apply_mask(snr, mask)

models = ['Tensor', 'BallStick_r1', 'NODDI', 'CHARMED_r1']
model_maps = {'BallStick_r1': 'FS.std',
              'Tensor': 'Tensor.FA.std',
              'NODDI': 'w_ic.w.std',
              'CHARMED_r1': 'FR.std', }


def scatter_plot(ax, samples_x, samples_y, colors):
    ax.scatter(samples_x, samples_y, c=colors)

    ax.set_xlim(np.min(samples_x)*0.95, np.max(samples_x)*1.06)
    ax.set_ylim(np.min(samples_y)*0.95, np.max(samples_y)*1.05)

    ax.xaxis.offsetText.set_visible(False)
    ax.yaxis.offsetText.set_visible(False)


# def reject_outliers(data, m = 2.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/(mdev if mdev else 1.)
#     return data[s<m]
# def reject_outliers(data, m=2.):
#     return data[abs(data - np.mean(data)) < m * np.std(data)]
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# def cleanup(opt, sample, snr_roi):
#     bin_width = 2.5
#     x_ind = list(np.arange(bin_width, 50, bin_width))
#
#     indices = []
#
#     for bin in x_ind:
#         snr_indices = np.where(np.logical_and(bin - bin_width / 2 <= snr_roi, snr_roi <= bin + bin_width / 2))
#
#         opt_good = np.where(abs(opt[snr_indices] - np.mean(opt[snr_indices])) < 2 * np.std(opt[snr_indices]))
#         sample_good = np.where(abs(sample[snr_indices] - np.mean(sample[snr_indices])) < 2 * np.std(sample[snr_indices]))
#
#         indices.append(np.intersect1d(opt_good, sample_good))
#
#     indices = np.array(indices).flatten()
#
#     return opt[indices], sample[indices], snr_roi[indices]




def plot_summary(ax, snr_roi, stats, color):
    means = []
    stds = []
    x_ind = list(np.arange(5, 50, 2.5))

    bin_width = 2.5
    for bin in x_ind:
        values = stats[np.where(np.logical_and(bin - bin_width / 2 <= snr_roi, snr_roi <= bin + bin_width / 2))]
        values = values[~is_outlier(values)]
        means.append(np.mean(values))
        stds.append(np.std(values))

    ax.errorbar(x_ind, means, yerr=stds, c=color)


final_mask = input_data.mask * mdt.load_brain_mask(wm_mask)

fig, axii = plt.subplots(2, 2)
axii = axii.reshape(-1)

for ind, model_name in enumerate(models):
    # snr = gaussian_filter(snr, sigma=0.6)
    # maps = {k: gaussian_filter(v, sigma=0.6) for k, v in maps.items()}

    opt = mdt.create_roi(output_pjoin(model_name, model_maps[model_name]), final_mask)
    sample = mdt.create_roi(output_pjoin(model_name, 'samples', 'model_defined_maps', model_maps[model_name]), final_mask)
    snr_roi = mdt.create_roi(snr, final_mask)

    # opt, sample, snr_roi = cleanup(opt, sample, snr_roi)

    ax = axii[ind]
    scatter_plot(ax, snr_roi, opt, '#c45054')
    scatter_plot(ax, snr_roi, sample, '#0773b3')

    plot_summary(ax, snr_roi, opt, '#e79f27')
    plot_summary(ax, snr_roi, sample, '#65e065')

    ax.set_xlabel('SNR')
    ax.set_ylabel('Mode and Mean')
    ax.set_title(model_name + ' ' + model_maps[model_name])

fig.suptitle('SNR vs Mode and Mean')
plt.show()

