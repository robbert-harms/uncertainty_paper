import mdt
import numpy as np


__author__ = 'Robbert Harms'
__date__ = "2018-08-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/single_voxel_simulation/')
simulations_unweighted_signal_height = 1e4

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

protocols = [
    'hcp_mgh_1003',
    # 'rheinland_v3a_1_2mm'
]

noise_snrs = [2, 5, 10, 20, 30, 40, 50]
# noise_snrs = [50]

model_names = [
    # 'BallStick_r1',
    # 'BallStick_r2',
    # 'BallStick_r3',
    # 'NODDI',
    'BinghamNODDI_r1',
    # 'Tensor',
    # 'CHARMED_r1',
    # 'CHARMED_r2',
    # 'CHARMED_r3',
]


class BinghamNODDI_r1(mdt.get_template('composite_models', 'BinghamNODDI_r1')):
    likelihood_function = 'Gaussian'


for model_name in model_names:
    for protocol_name in protocols:
        for snr in noise_snrs:
            noise_std = simulations_unweighted_signal_height / snr
            current_pjoin = pjoin.create_extended(protocol_name, model_name)

            input_data = mdt.load_input_data(
                current_pjoin('noisy_signals_{}.nii'.format(snr)),
                pjoin(protocol_name + '.prtcl'),
                current_pjoin('mask.nii'),
                noise_std=noise_std)

            print('Going to process', model_name, protocol_name, snr)

            fit_results = mdt.fit_model(
                model_name,
                input_data,
                current_pjoin('output', str(snr)),
                post_processing={'uncertainties': True}
            )

            mdt.sample_model(
                model_name,
                input_data,
                current_pjoin('output', str(snr)),
                nmr_samples=nmr_samples[model_name],
                burnin=1000,
                thinning=0,
                initialization_data={'inits': fit_results},
                store_samples=True
            )

            free_param_names = mdt.get_model(model_name)().get_free_param_names()
            original_parameters = mdt.load_volume_maps(current_pjoin())['original_parameters']
            mdt.compute_fim(
                model_name,
                input_data,
                dict(zip(free_param_names, np.split(original_parameters, len(free_param_names), axis=3))),
                current_pjoin('output', str(snr), 'CRLB')
            )