import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage.filters import gaussian_filter
import mdt


input_path = r'/home/robbert/programming/python/uncertainty_paper/data/single_slice/'
output_path = r'/home/robbert/phd-data/papers/uncertainty_paper/single_slice/'


model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'NODDI',
    'BinghamNODDI_r1',
    'Tensor',
    'CHARMED_r1',
]


def get_mle_std(output_dir, model_name):
    maps = mdt.load_volume_maps(output_dir)

    if model_name == 'Tensor':
        return maps['Tensor.FA.std']
    elif model_name == 'BallStick_r1':
        return maps['w_stick0.w.std']
    elif model_name == 'BallStick_r2':
        # return maps['FS.std']
        return maps['w_stick0.w.std']
    elif model_name == 'BallStick_r3':
        # return maps['FS.std']
        return maps['w_stick0.w.std']
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


def get_inverse_snr_map(subject_info):
    input_data = subject_info.get_input_data()

    unweighted_volumes = input_data.protocol.get_unweighted_indices()

    snr = np.mean(input_data.signal4d[..., unweighted_volumes], axis=-1) \
          / np.std(input_data.signal4d[..., unweighted_volumes], axis=-1)

    return {'snr': snr, 'inverse_snr': 1/snr, 'mask': input_data.mask}


snr_maps = mdt.batch_apply(input_path, get_inverse_snr_map)
outputs = mdt.run_function_on_batch_fit_output(
    lambda output_info: get_mle_std(output_info.output_path, output_info.model_name),
    output_path,
    model_names=model_names)


results = {
    'mgh': snr_maps['mgh'],
    'rheinland': snr_maps['rheinland']
}

for data_name in outputs:
    for map_name in outputs[data_name]:
        if map_name == 'Tensor':
            results[data_name][map_name] = np.clip(outputs[data_name][map_name], 0, 0.15)
        else:
            results[data_name][map_name] = np.clip(outputs[data_name][map_name], 0, 0.05)

for data_name in results:
    mask = binary_erosion(results[data_name]['mask'][..., 0], iterations=2)[..., None]
    results[data_name] = mdt.apply_mask(results[data_name], mask)
    results[data_name] = {k: gaussian_filter(v, sigma=1) for k, v in results[data_name].items()}

# width: 1371
# height: 804

mdt.view_maps(results['mgh'], config='''
annotations: []
colorbar_settings: {location: null, nmr_ticks: 3, power_limits: null, round_precision: null,
  visible: null}
colormap: hot
colormap_masked_color: null
dimension: 2
flipud: false
font: {family: sans-serif, size: 20}
grid_layout:
- Rectangular
- cols: null
  rows: 2
  spacings: {bottom: 0.03, hspace: 0.2, left: 0.01, right: 0.92, top: 0.94, wspace: 0.5}
interpolation: bilinear
map_plot_options:
  BallStick_r1:
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
    scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
    show_title: null
    title: 'BallStick_in1 -

      Stick0.w (std.)'
    title_spacing: null
  BallStick_r2:
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
    scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
    show_title: null
    title: 'BallStick_in2 -

      Stick0.w (std.)'
    title_spacing: null
  BallStick_r3:
    clipping: {use_max: true, use_min: true, vmax: 0.028, vmin: 0.0}
    colorbar_label: null
    colorbar_settings:
      location: null
      nmr_ticks: null
      power_limits: [-2, 5]
      round_precision: null
      visible: null
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
    show_title: null
    title: 'BallStick_in3 -

      Stick0.w (std.)'
    title_spacing: null
  BinghamNODDI_r1:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings:
      location: null
      nmr_ticks: null
      power_limits: [-2, 5]
      round_precision: null
      visible: null
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.04, vmin: 0.0}
    show_title: null
    title: 'Bingham-NODDI -

      FR (std.)'
    title_spacing: null
  CHARMED_r1:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings:
      location: null
      nmr_ticks: null
      power_limits: [-2, 5]
      round_precision: null
      visible: null
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.04, vmin: 0.0}
    show_title: null
    title: 'CHARMED_in1 -

      FR (std.)'
    title_spacing: null
  NODDI:
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
    scale: {use_max: true, use_min: true, vmax: 0.04, vmin: 0.0}
    show_title: null
    title: 'NODDI

      (FR)'
    title_spacing: null
  Tensor:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: null, nmr_ticks: null, power_limits: null, round_precision: null,
      visible: null}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.12, vmin: 0.0}
    show_title: null
    title: 'Tensor -

      FA (std.)'
    title_spacing: null
  inverse_snr:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: null, nmr_ticks: null, power_limits: null, round_precision: null,
      visible: null}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 0.15, vmin: 0.0}
    show_title: null
    title: 1 / SNR
    title_spacing: null
  snr:
    clipping: {use_max: false, use_min: false, vmax: 0.0, vmin: 0.0}
    colorbar_label: null
    colorbar_settings: {location: null, nmr_ticks: null, power_limits: null, round_precision: null,
      visible: null}
    colormap: null
    colormap_masked_color: null
    colormap_order: null
    colormap_weight_map: null
    interpret_as_colormap: false
    mask_name: null
    scale: {use_max: true, use_min: true, vmax: 30.0, vmin: 0.0}
    show_title: null
    title: SNR
    title_spacing: null
maps_to_show: [snr, BallStick_r1, BallStick_r2, BallStick_r3, inverse_snr, BinghamNODDI_r1,
  CHARMED_r1, Tensor]
mask_name: null
rotate: 270
show_axis: false
show_titles: true
slice_index: 0
title: null
title_spacing: 0.03
volume_index: 0
zoom:
  p0: {x: 19, y: 17}
  p1: {x: 117, y: 128}

''')

#
# mdt.view_maps(results['rheinland'], config='''
# colorbar_settings:
#   location: right
#   nmr_ticks: 4
#   power_limits: [-2, 5]
#   round_precision: 3
#   visible: true
# font: {family: sans-serif, size: 20}
# grid_layout:
# - Rectangular
# - cols: null
#   rows: 1
#   spacings: {bottom: 0.03, hspace: 0.2, left: 0.01, right: 0.92, top: 0.94, wspace: 1.0}
# map_plot_options:
#   BallStick_r1:
#     scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
#     title: BallStick_in1
#     title_spacing: 0.2
#   BallStick_r2:
#     scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
#     title: BallStick_in2
#     title_spacing: 0.2
#   BallStick_r3:
#     scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
#     title: BallStick_in3
#     title_spacing: 0.2
#   BinghamNODDI_r1:
#     scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
#     title: Bingham-NODDI
#     title_spacing: 0.2
#   CHARMED_r1:
#     scale: {use_max: true, use_min: true, vmax: 0.03, vmin: 0.0}
#     title: CHARMED_in1
#     title_spacing: 0.2
#   inverse_snr:
#     scale: {use_max: true, use_min: true, vmax: 0.15, vmin: 0.0}
#     title: 1 / SNR
#     title_spacing: 0.2
# maps_to_show: [inverse_snr, BallStick_r1, BallStick_r2, BallStick_r3, BinghamNODDI_r1,
#   CHARMED_r1]
# title_spacing: 0.03
# zoom:
#   p0: {x: 17, y: 4}
#   p1: {x: 89, y: 99}
#
# ''')
