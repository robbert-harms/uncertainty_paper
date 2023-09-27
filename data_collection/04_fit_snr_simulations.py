import mdt

__author__ = 'Robbert Harms'
__date__ = "2018-08-14"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"

pjoin = mdt.make_path_joiner('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/')
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
    'rheinland_v3a_1_2mm'
]

noise_snrs = [2, 5, 10, 20, 30, 40, 50]

model_names = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'NODDI',
    'BinghamNODDI_r1',
    'Tensor',
    'CHARMED_r1',
]


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

            for trial_ind in range(nmr_trials):
                print('Going to process', trial_ind, model_name, protocol_name, snr)

                fit_results = mdt.fit_model(
                    model_name,
                    input_data,
                    current_pjoin('output', str(snr), str(trial_ind)))

                mdt.sample_model(
                    model_name,
                    input_data,
                    current_pjoin('output', str(snr), str(trial_ind)),
                    nmr_samples=nmr_samples[model_name],
                    burnin=1000,
                    thinning=0,
                    initialization_data={'inits': fit_results},
                    store_samples=False
                )