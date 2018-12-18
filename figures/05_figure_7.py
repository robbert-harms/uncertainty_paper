import mdt
import os
import numpy as np

__author__ = 'Robbert Harms'
__date__ = '2018-12-07'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


registration_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/registration/')

maps_to_process = {
    # 'Tensor': ['Tensor.FA', 'Tensor.FA.std'],
    # 'NODDI': ['w_ic.w', 'w_ic.w.std'],
    'BinghamNODDI_r1': ['w_in0.w', 'w_in0.w.std'],
    # 'CHARMED_r1': ['FR', 'FR.std'],
    # 'BallStick_r1': ['FS', 'FS.std'],
    # 'BallStick_r2': ['FS', 'FS.std'],
    # 'BallStick_r3': ['FS', 'FS.std']
}


def get_subject_maps(model_name, map_name):
    data_name = '{}_{}'.format(model_name, map_name)
    map_list = []
    for subject in os.listdir(registration_pjoin()):
        data = mdt.load_nifti(registration_pjoin(subject, 'warped_' + data_name)).get_data()
        map_list.append(data)
    return map_list


def regular_average():
    result = {}
    for model_name, maps in maps_to_process.items():
        map_name = maps[0]
        data_name = '{}_{}'.format(model_name, map_name)

        subject_volumes = np.stack(get_subject_maps(model_name, map_name), axis=-1)

        result[data_name] = np.mean(subject_volumes, axis=-1)
        result[data_name + '.std'] = np.std(subject_volumes, axis=-1)

    return result


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
    return average, np.sqrt(variance)


def weighted_average():
    result = {}
    for model_name, maps in maps_to_process.items():
        subject_volumes = np.stack(get_subject_maps(model_name, maps[0]), axis=-1)
        subject_volumes_stds = np.stack(get_subject_maps(model_name, maps[1]), axis=-1)

        stds_square = subject_volumes_stds**2
        stds_square[stds_square < 1e-4] = 1e-4

        weights = 1. / stds_square

        average, stds = _weighted_avg_and_std(subject_volumes, weights)

        data_name = '{}_{}'.format(model_name, maps[0])
        result[data_name] = average
        result[data_name + '.std'] = stds
    return result


all_maps = {}
for key, value in regular_average().items():
    all_maps['regular_' + key] = np.squeeze(value)

for key, value in weighted_average().items():
    all_maps['weighted_' + key] = np.squeeze(value)

for model_name, maps in maps_to_process.items():
    all_maps['point_diff_' + model_name] = \
        all_maps['regular_{}_{}'.format(model_name, maps[0])] - all_maps['weighted_{}_{}'.format(model_name, maps[0])]

    all_maps['std_diff_' + model_name] = \
        all_maps['regular_{}_{}'.format(model_name, maps[1])] - all_maps['weighted_{}_{}'.format(model_name, maps[1])]

    all_maps['weighted_diff_smaller_' + model_name] = \
        all_maps['regular_{}_{}'.format(model_name, maps[1])] >= all_maps['weighted_{}_{}'.format(model_name, maps[1])]

    all_maps['weighted_diff_smaller_' + model_name] = all_maps['weighted_diff_smaller_' + model_name].astype(np.int16)

    all_maps['weighted_diff_smaller_' + model_name][
        all_maps['regular_{}_{}'.format(model_name, maps[1])]
        < all_maps['weighted_{}_{}'.format(model_name, maps[1])]] = -1


mdt.apply_mask(all_maps, '/usr/share/data/fsl-mni152-templates/FMRIB58_FA_1mm.nii.gz')
mdt.view_maps(
    all_maps,
    # save_filename='/tmp/uncertainty_paper/bingham_noddi.png',
    config='''
annotations:
- arrow_width: 1.0
  font_size: null
  marker_size: 1.0
  text_distance: 0.12
  text_location: upper left
  text_template: '{voxel_index}

    {value:.3g}'
  voxel_index: [77, 153, 90]
slice_index: 90
zoom:
  p0: {x: 22, y: 17}
  p1: {x: 156, y: 192}
''')
