import os
from subprocess import call
import shutil
from textwrap import dedent

import mdt


__author__ = 'Robbert Harms'
__date__ = '2018-12-06'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


opt_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/hcp_mgh_output/')
output_pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/registration/')
fa_reference = '/usr/share/data/fsl-mni152-templates/FMRIB58_FA_1mm.nii.gz'
max_nmr_subjects = -1

maps_to_process = {
    'Tensor': ['Tensor.FA', 'Tensor.FA.std'],
    'NODDI': ['w_ic.w', 'w_ic.w.std'],
    'BinghamNODDI_r1': ['w_in0.w', 'w_in0.w.std'],
    'CHARMED_r1': ['FR', 'FR.std'],
    'BallStick_r1': ['FS', 'FS.std'],
    'BallStick_r2': ['FS', 'FS.std'],
    'BallStick_r3': ['FS', 'FS.std']
}


def copy_maps(results_pjoin, output_pjoin):
    subjects = sorted(os.listdir(results_pjoin()))[:max_nmr_subjects]

    for subject in subjects:
        for model_name, maps in maps_to_process.items():
            for map_name in maps:
                out_pjoin = output_pjoin.create_extended(subject, make_dirs=True)
                shutil.copy(results_pjoin(subject, model_name, map_name + '.nii.gz'),
                            out_pjoin('{}_{}.nii.gz'.format(model_name, map_name)))


def correct_mgh_image_position(input_fname, output_fname=None):
    """The HCP MGH data is ill-positioned for the registration algorithm, this function corrects that."""
    header = mdt.load_nifti(input_fname).get_header()
    data = mdt.load_nifti(input_fname).get_data()

    if output_fname is None:
        output_fname = input_fname

    mdt.write_nifti(data[:, ::-1], output_fname, header)


def correct_mgh_maps(output_pjoin):
    for subject in os.listdir(output_pjoin()):
        for img in os.listdir(output_pjoin(subject)):
            correct_mgh_image_position(output_pjoin(subject, img))


def get_registration_commands(output_pjoin):
    """
    This outputs shell scripts which should all be run, i.e. by using:

        parallel -j0 bash :::: <(ls *.sh)

    After that, remove the shell scripts.
    """
    for subject in os.listdir(output_pjoin()):
        commands = dedent('''
            cd {subject_dir}
            
            flirt -ref {template} -in {tensor_map} -omat affine.mat
            fnirt --in={tensor_map} --aff=affine.mat --cout=nonlin_transf --config=FA_2_FMRIB58_1mm
        '''.format(subject_dir=output_pjoin(subject), tensor_map='Tensor_Tensor.FA.nii.gz', template=fa_reference))

        for model_name, maps in maps_to_process.items():
            for map_name in maps:
                commands += dedent('''
                    applywarp --ref={template} --in={map_name} --warp=nonlin_transf --out=warped_{map_name}
                '''.format(template=fa_reference, map_name=model_name + '_' + map_name + '.nii.gz'))

        with open(output_pjoin(subject + '_run.sh'), 'w') as f:
            f.writelines(commands)


def check_registration():
    map_configs = ''
    fa_maps = {}
    for subject in os.listdir(output_pjoin()):
        fa_maps[subject] = mdt.load_nifti(output_pjoin(subject, 'warped_Tensor_Tensor.FA.nii.gz')).get_data()
        # map_configs += '''
        # {}:
        #     scale: {{use_max: true, use_min: true, vmax: 0.1, vmin: 0.0}}
        # '''.format(subject)

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
    maps_to_show: [mgh_1003, mgh_1019, mgh_1015, mgh_1031, mgh_1023, mgh_1029, mgh_1002,
      mgh_1007, mgh_1027, mgh_1022, mgh_1024, mgh_1011, mgh_1034, mgh_1017, mgh_1026,
      mgh_1010, mgh_1009, mgh_1004, mgh_1018, mgh_1013, mgh_1020, mgh_1030, mgh_1005,
      mgh_1032, mgh_1014, mgh_1006, mgh_1025, mgh_1008, mgh_1028, mgh_1012, mgh_1033,
      mgh_1021, mgh_1001, mgh_1016]
    slice_index: 90
    zoom:
      p0: {x: 16, y: 14}
      p1: {x: 161, y: 200}
    
    '''
    if map_configs:
        config += '''
    map_plot_options:
    ''' + map_configs + '''
    '''

    mdt.view_maps(fa_maps, config=config)


# copy_maps(opt_pjoin, output_pjoin)
# correct_mgh_maps(output_pjoin)
# get_registration_commands(output_pjoin)
check_registration()