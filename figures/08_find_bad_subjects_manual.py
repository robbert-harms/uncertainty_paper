import os
import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-12-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


output_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/registration/')

def check_registration():
    map_configs = ''
    maps = {}

    particularly_bad_subjects = [
        'mgh_1002',
        'mgh_1004',
        'mgh_1008',
        'mgh_1009',
        'mgh_1012',
        'mgh_1013',
        'mgh_1015',
        'mgh_1017',
        'mgh_1021',
        'mgh_1022',
        'mgh_1032'
    ]

    for subject in os.listdir(output_pjoin()):
    # for subject in particularly_bad_subjects:
        point_map = mdt.load_nifti(output_pjoin(subject, 'warped_BinghamNODDI_r1_w_in0.w')).get_data()
        std_map = mdt.load_nifti(output_pjoin(subject, 'warped_BinghamNODDI_r1_w_in0.w.std')).get_data()

        maps[subject + '.std'] = std_map
        maps[subject] = point_map
        map_configs += '''
            {0}:
                scale: {{use_max: true, use_min: true, vmax: 0.8, vmin: 0.0}}
            {0}.std:
                scale: {{use_max: true, use_min: true, vmax: 0.1, vmin: 0.0}}
        '''.format(subject)

    config = '''
        colorbar_settings:
          location: right
          nmr_ticks: 4
          power_limits: [-3, 4]
          round_precision: 3
          visible: false
        grid_layout:
        - Rectangular
        - cols: null
          rows: 4
          spacings: {bottom: 0.03, hspace: 0.15, left: 0.1, right: 0.86, top: 0.97, wspace: 0.4}
        slice_index: 90
        zoom:
          p0: {x: 16, y: 14}
          p1: {x: 161, y: 200}
        colormap_masked_color: 'k'

    '''
    if map_configs:
        config += '''
        map_plot_options:
        ''' + map_configs + '''
    '''

    config += '''
        maps_to_show: [''' + ', '.join(sorted(maps)) + '''] 
    '''

    mdt.view_maps(maps, config=config)


check_registration()