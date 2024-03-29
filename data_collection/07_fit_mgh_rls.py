import glob
import mdt
import os
from mdt.lib.batch_utils import SimpleBatchProfile, BatchFitProtocolLoader, SimpleSubjectInfo
from mdt.lib.masking import generate_simple_wm_mask

__author__ = 'Robbert Harms'
__date__ = '2018-11-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'NODDI',
    'Tensor',
    'BinghamNODDI_r1',
    'CHARMED_r1'
]


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


'''batch fit all subjects'''
mdt.batch_fit('/home/robbert/phd-data/rheinland/', model_names,
              batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
              subjects_selection=range(10))

mdt.batch_fit('/home/robbert/phd-data/hcp_mgh/', model_names,
              batch_profile='HCP_MGH', subjects_selection=range(10))


def func(subject_info):
    subject_id = subject_info.subject_id

    wm_mask = generate_simple_wm_mask(
        os.path.join(subject_info.data_folder[:-1] + '_output', subject_id, 'Tensor', 'Tensor.FA.nii.gz'),
        subject_info.get_input_data().mask,
        threshold=0.3,
        median_radius=3,
        nmr_filter_passes=4)
    mdt.write_nifti(wm_mask, os.path.join(subject_info.data_folder[:-1] + '_output', subject_id, 'wm_mask'))


'''generate wm mask for all subjects'''
mdt.batch_apply('/home/robbert/phd-data/rheinland/', func,
                batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
                subjects_selection=range(10))

mdt.batch_apply('/home/robbert/phd-data/hcp_mgh/', func,
                batch_profile='HCP_MGH', subjects_selection=range(10))



nmr_samples = {
    'BallStick_r1': 11000,
    'BallStick_r2': 15000,
    'BallStick_r3': 25000,
    'NODDI': 15000,
    'BinghamNODDI_r1': 20000,
    'Tensor': 13000,
    'CHARMED_r1': 17000,
}


def mcmc_sample(subject_info):
    output_folder = subject_info.data_folder[:-1] + '_output'
    subject_id = subject_info.subject_id

    for model_name in model_names:
        print(subject_id, model_name)

        starting_point = mdt.fit_model(model_name,
                                       subject_info.get_input_data(),
                                       output_folder + '/' + subject_id)

        with mdt.config_context('''
            processing_strategies:
                sampling:
                    max_nmr_voxels: 5000
        '''):
            mdt.sample_model(model_name,
                             subject_info.get_input_data(),
                             output_folder + '/' + subject_id,
                             nmr_samples=nmr_samples[model_name],
                             initialization_data={'inits': starting_point},
                             store_samples=False)


'''for first 10 subjects, also do sampling'''
mdt.batch_apply('/home/robbert/phd-data/rheinland/', mcmc_sample,
                batch_profile=RheinLandBatchProfile(resolutions_to_use=['data_ms20']),
                subjects_selection=range(10))

mdt.batch_apply('/home/robbert/phd-data/hcp_mgh/', mcmc_sample,
                batch_profile='HCP_MGH',
                subjects_selection=range(9, 10))

