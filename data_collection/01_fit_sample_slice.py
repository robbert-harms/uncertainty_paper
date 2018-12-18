import mdt

__author__ = 'Robbert Harms'
__date__ = '2018-11-01'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


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
    'CHARMED_r2',
    'CHARMED_r3'
]

nmr_samples = {
    'BallStick_r1': 11000,
    'BallStick_r2': 15000,
    'BallStick_r3': 25000,
    'NODDI': 15000,
    'BinghamNODDI_r1': 20000,
    'Tensor': 13000,
    'CHARMED_r1': 17000,
    'CHARMED_r2': 25000,
    'CHARMED_r3': 25000
}


def sample_subject(subject_info):
    for model_name in model_names:
        starting_point = mdt.fit_model(model_name,
                                       subject_info.get_input_data(),
                                       output_path + '/' + subject_info.subject_id)

        mdt.sample_model(model_name,
                         subject_info.get_input_data(),
                         output_path + '/' + subject_info.subject_id,
                         nmr_samples=nmr_samples[model_name],
                         initialization_data={'inits': starting_point},
                         store_samples=True)


mdt.batch_apply(sample_subject, input_path)