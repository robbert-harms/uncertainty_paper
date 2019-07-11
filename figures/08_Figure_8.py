from scipy.ndimage import binary_erosion

import mdt
import os
import numpy as np

from mdt.lib.masking import generate_simple_wm_mask

__author__ = 'Robbert Harms'
__date__ = '2018-12-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


registration_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/registration/')

mask = mdt.load_brain_mask('/usr/share/data/fsl-mni152-templates/FMRIB58_FA_1mm.nii.gz')
mask = binary_erosion(mask, iterations=1)

subjects_to_filter = [
    # 'mgh_1002',
    # 'mgh_1004',
        'mgh_1008',
        'mgh_1009',
    # 'mgh_1012',
        'mgh_1013',
    # 'mgh_1015',
        'mgh_1017',
    # 'mgh_1021',
    # 'mgh_1022',
        'mgh_1032'
]

# subjects_to_filter = []

map_names = {
    'Tensor': 'Tensor.FA',
    'NODDI': 'w_ic.w',
    'BinghamNODDI_r1': 'w_in0.w',
    'CHARMED_r1': 'FR',
    'BallStick_r1': 'FS',
    'BallStick_r2': 'FS',
    'BallStick_r3': 'FS'
}


def _get_subject_maps(model_name, map_name):
    data_name = '{}_{}'.format(model_name, map_name)
    map_list = []
    for subject in os.listdir(registration_pjoin()):
        if subject in subjects_to_filter:
            continue
        data = mdt.load_nifti(registration_pjoin(subject, 'warped_' + data_name)).get_data()
        map_list.append(data)
    return map_list


def _weighted_avg_and_std(values, weights):
    """Return the weighted average and standard deviation.

    Args:
        values (ndarray): the matrix with values, taking the average over the last axis
        weights (ndarray): the weights corresponding to every value

    Returns:
        tuple[ndarray, ndarray]: the weighted average and weighted std.
    """
    average = np.average(values, weights=weights, axis=-1)[..., None]
    variance = np.average((values-average)**2, weights=weights, axis=-1)
    return average[..., 0], np.sqrt(variance)


def regular_average(model_name):
    subject_volumes = np.stack(_get_subject_maps(model_name, map_names[model_name]), axis=-1)
    return np.mean(subject_volumes, axis=-1), np.std(subject_volumes, axis=-1)


def weighted_average(model_name):
    map_name = map_names[model_name]
    subject_volumes = np.stack(_get_subject_maps(model_name, map_name), axis=-1)
    subject_volumes_stds = np.stack(_get_subject_maps(model_name, map_name + '.std'), axis=-1)

    stds_square = subject_volumes_stds**2
    stds_square[stds_square < 1e-4] = 1e-4

    weights = 1. / stds_square

    return _weighted_avg_and_std(subject_volumes, weights)


fa_average, _ = regular_average('BallStick_r1')
wm_mask = generate_simple_wm_mask(fa_average, mask, threshold=0.3, median_radius=2, nmr_filter_passes=1)
wm_mask = binary_erosion(wm_mask, iterations=2)

# mdt.view_maps({'wm_mask': wm_mask, 'data': fa_average}, config='''
# maps_to_show: [data, wm_mask]
# slice_index: 90
# ''')
# exit()

model_name = 'BinghamNODDI_r1'
map_name = map_names[model_name]

all_maps = {}

reg_mean, reg_std = regular_average(model_name)
all_maps['regular_{}_{}'.format(model_name, map_name)] = reg_mean
all_maps['regular_{}_{}.std'.format(model_name, map_name)] = reg_std

wgh_mean, wgh_std = weighted_average(model_name)
all_maps['weighted_{}_{}'.format(model_name, map_name)] = wgh_mean
all_maps['weighted_{}_{}.std'.format(model_name, map_name)] = wgh_std

all_maps['point_diff_' + model_name] = 100 * (wgh_mean - reg_mean) / reg_mean
all_maps['std_diff_' + model_name] = 100 * (wgh_std - reg_std) / reg_std

mdt.apply_mask(all_maps, mask)


all_maps['std_diff_' + model_name] = np.ma.masked_where(wm_mask < 1, all_maps['std_diff_' + model_name])
all_maps['point_diff_' + model_name] = np.ma.masked_where(wm_mask < 1, all_maps['point_diff_' + model_name])


mdt.view_maps(
    all_maps,
    # save_filename='/tmp/uncertainty_paper/bingham_noddi.png',
    config='''
annotations:
- arrow_width: 1.0
  font_size: null
  marker_size: 3.0
  text_distance: 0.08
  text_location: upper left
  text_template: '{value:.2g}'
  voxel_index: [60, 149, 90]
colorbar_settings:
  location: right
  nmr_ticks: 4
  power_limits: [-3, 4]
  round_precision: 3
  visible: true
colormap: hot
colormap_masked_color: k
dimension: 2
flipud: false
font: {family: sans-serif, size: 21}
grid_layout:
- Rectangular
- cols: null
  rows: 3
  spacings: {bottom: 0.03, hspace: 0.23, left: 0.1, right: 0.86, top: 0.97, wspace: 0}
interpolation: bilinear
map_plot_options:
  point_diff_BinghamNODDI_r1:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: null}
    colormap: BrBG_r
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 10.0, vmin: -10.0}
    show_title: null
    title: '% difference mean'
    title_spacing: null
  regular_BinghamNODDI_r1_w_in0.w:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: null, nmr_ticks: null, power_limits: null, round_precision: null,
      visible: false}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.8, vmin: 0.2}
    show_title: null
    title: Regular mean
    title_spacing: null
  regular_BinghamNODDI_r1_w_in0.w.std:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: null, nmr_ticks: null, power_limits: null, round_precision: null,
      visible: false}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.15, vmin: 0.0}
    show_title: null
    title: Regular std.
    title_spacing: null
  std_diff_BinghamNODDI_r1:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: null}
    colormap: BrBG_r
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 20.0, vmin: -20.0}
    show_title: null
    title: '% difference std.'
    title_spacing: null
  weighted_BinghamNODDI_r1_w_in0.w:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: null}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.8, vmin: 0.2}
    show_title: null
    title: Weighted mean
    title_spacing: null
  weighted_BinghamNODDI_r1_w_in0.w.std:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: null}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.15, vmin: 0.0}
    show_title: null
    title: Weighted std.
    title_spacing: null
maps_to_show: [regular_BinghamNODDI_r1_w_in0.w, regular_BinghamNODDI_r1_w_in0.w.std,
  weighted_BinghamNODDI_r1_w_in0.w, weighted_BinghamNODDI_r1_w_in0.w.std, point_diff_BinghamNODDI_r1,
  std_diff_BinghamNODDI_r1]
mask_name: null
rotate: 90
show_axis: false
show_titles: true
slice_index: 90
title: null
title_spacing: null
volume_index: 0
zoom:
  p0: {x: 22, y: 24}
  p1: {x: 156, y: 192}
''')
