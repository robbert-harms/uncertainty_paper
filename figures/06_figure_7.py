from scipy.ndimage import binary_erosion

import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-12-20'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'

output_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/registration/diff_eddy/')
mask = mdt.load_brain_mask('/home/robbert/Downloads/fsl/data/standard/FMRIB58_FA_1mm.nii.gz')
mask = binary_erosion(mask, iterations=1)

maps = {}

subjects_to_load = [
    'mgh_1005',
    'mgh_1016',
    'mgh_1017'
]

for subject in subjects_to_load:
    point_map = mdt.load_nifti(output_pjoin(subject, 'warped_BinghamNODDI_r1_w_in0.w')).get_data()
    std_map = mdt.load_nifti(output_pjoin(subject, 'warped_BinghamNODDI_r1_w_in0.w.std')).get_data()

    maps[subject + '.std'] = std_map
    maps[subject] = point_map

mdt.apply_mask(maps, mask)

# height : 1000px

mdt.view_maps(maps, config='''
colorbar_settings:
  location: right
  nmr_ticks: 4
  power_limits: [-3, 4]
  round_precision: 3
  visible: false
colormap_masked_color: k
font: {family: sans-serif, size: 21}
grid_layout:
- Rectangular
- cols: 2
  rows: null
  spacings: {bottom: 0.03, hspace: 0.1, left: 0.1, right: 0.86, top: 0.97, wspace: 0.1}
map_plot_options:
  mgh_1005:
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: true}
    scale: {use_max: true, use_min: true, vmax: 0.9, vmin: 0.1}
    title: ' '
  mgh_1005.std:
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: true}
    clipping: {use_max: true, use_min: true, vmax: 0.045, vmin: 0.0}
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
  mgh_1016:
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: false}
    scale: {use_max: true, use_min: true, vmax: 0.9, vmin: 0.1}
    title: ' '
  mgh_1016.std:
    clipping: {use_max: true, use_min: true, vmax: 0.04, vmin: 0.0}
    colorbar_settings: {location: bottom, nmr_ticks: 3, power_limits: null, round_precision: null,
      visible: false}
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: ' '
  mgh_1017:
    scale: {use_max: true, use_min: true, vmax: 0.9, vmin: 0.1}
    title: Mean
  mgh_1017.std:
    clipping: {use_max: true, use_min: true, vmax: 0.045, vmin: 0.0}
    scale: {use_max: true, use_min: true, vmax: 0.05, vmin: 0.0}
    title: std.
maps_to_show: [mgh_1017, mgh_1017.std, mgh_1016, mgh_1016.std, mgh_1005, mgh_1005.std]
slice_index: 90
zoom:
  p0: {x: 23, y: 25}
  p1: {x: 155, y: 191}
''')
