import numpy as np

import mdt

__author__ = 'Robbert Harms'
__date__ = "2018-08-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

base_path = '/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/'

pjoin = mdt.make_path_joiner(base_path)
nmr_trials = 10
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

noise_snrs = [
    2,
    5,
    10,
    20,
    30,
    40,
    50
]

model_names = [
    # 'BallStick_r1',
    # 'BallStick_r2',
    # 'BallStick_r3',
    # 'NODDI',
    'BinghamNODDI_r1',
    # 'Tensor',
    # 'CHARMED_r1',
]


for model_name in model_names:
    for protocol_name in protocols:
        current_pjoin = pjoin.create_extended(protocol_name, model_name)

        for snr in noise_snrs:
            noise_std = simulations_unweighted_signal_height / snr

            input_data = mdt.load_input_data(
                current_pjoin('noisy_signals_{}.nii'.format(snr)),
                pjoin(protocol_name + '.prtcl'),
                current_pjoin('mask.nii'),
                noise_std=noise_std)

            original_parameters = mdt.load_nifti(current_pjoin('original_parameters')).get_data()
            free_param_names = mdt.get_model(model_name)().get_free_param_names()

            # model = mdt.get_model(model_name)()
            # model.set_input_data(input_data)
            # extra_data = model.get_post_optimization_output(
            #     np.reshape(original_parameters, (original_parameters.shape[1], original_parameters.shape[-1])))
            # mdt.write_volume_maps(extra_data, current_pjoin('original_parameters_extra_output_maps', str(snr)))
            #
            # mdt.compute_fim(
            #     model_name,
            #     input_data,
            #     dict(zip(free_param_names, np.split(original_parameters, len(free_param_names), axis=3))),
            #     current_pjoin('CRLB', str(snr)),
            #     # cl_device_ind=1
            # )

            for trial_ind in range(1):
                print('Going to process', trial_ind, model_name, protocol_name, snr)

                fit_results = mdt.fit_model(
                    model_name,
                    input_data,
                    current_pjoin('output', str(snr), str(trial_ind)),
                    post_processing={'uncertainties': True},
                    recalculate=False,
                    # cl_device_ind=1
                )

                mdt.sample_model(
                    model_name,
                    input_data,
                    current_pjoin('output', str(snr), str(trial_ind)),
                    nmr_samples=nmr_samples[model_name],
                    burnin=1000,
                    thinning=0,
                    initialization_data={'inits': fit_results},
                    store_samples=False,
                    # cl_device_ind=1
                )
