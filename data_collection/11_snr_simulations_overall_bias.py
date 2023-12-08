from itertools import pairwise
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd


randomized_parameter_names = {
    'BallStick_r1': ['w_stick0.w'],
    'BallStick_r2': ['w_stick0.w', 'w_stick1.w'],
    'BallStick_r3': ['w_stick0.w', 'w_stick1.w', 'w_stick2.w'],
    'NODDI': ['w_ic.w', 'NODDI_IC.kappa', 'w_ec.w'],
    'BinghamNODDI_r1': ['w_in0.w', 'BinghamNODDI_IN0.k1', 'BinghamNODDI_IN0.kw', 'w_en0.w'],
    'CHARMED_r1': ['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'w_res0.w', 'CHARMEDRestricted0.d'],
    'Tensor': ['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1'],
}

additional_parameters = {
    'BallStick_r1': ['FS'],
    'BallStick_r2': ['FS'],
    'BallStick_r3': ['FS'],
    'CHARMED_r1': ['FR'],
    'Tensor': ['Tensor.FA'],
}

full_parameter_lists = {
    'BallStick_r1': ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi'],
    'BallStick_r2': ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi'],
    'BallStick_r3': ['S0.s0', 'w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi',
                     'w_stick2.w', 'Stick2.theta', 'Stick2.phi'],
    'NODDI': ['S0.s0', 'w_ic.w', 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa', 'w_ec.w'],
    'BinghamNODDI_r1': ['S0.s0', 'w_in0.w', 'BinghamNODDI_IN0.theta', 'BinghamNODDI_IN0.phi', 'BinghamNODDI_IN0.psi',
                        'BinghamNODDI_IN0.k1', 'BinghamNODDI_IN0.kw', 'w_en0.w'],
    'CHARMED_r1': ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi',
                   'w_res0.w', 'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi'],
    'CHARMED_r2': ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi',
                   'w_res0.w',
                   'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
                   'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi'],
    'CHARMED_r3': ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi',
                   'w_res0.w',
                   'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
                   'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi', 'w_res2.w',
                   'CHARMEDRestricted2.d', 'CHARMEDRestricted2.theta', 'CHARMEDRestricted2.phi'],
    'Tensor': ['S0.s0', 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi'],
}

protocols = [
    'hcp_mgh_1003',
    'rheinland_v3a_1_2mm'
]

models = [
    'BallStick_r1',
    'BallStick_r2',
    'BallStick_r3',
    'NODDI',
    'BinghamNODDI_r1',
    'Tensor',
    'CHARMED_r1',
]

noise_snrs = [2, 5, 10, 20, 30, 40, 50]

data_dir = Path('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/')


def compute_bias_mle(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    all_ground_truth_values = nib.load(str(data_dir / protocol / model / 'original_parameters.nii')).get_fdata()
    param_gt = all_ground_truth_values[..., full_parameter_lists[model].index(parameter)].flatten()

    mle = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                       / model / f'{parameter}.nii.gz')).get_fdata().flatten()

    return compute_bias(param_gt, mle)


def compute_bias_mcmc(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    all_ground_truth_values = nib.load(str(data_dir / protocol / model / 'original_parameters.nii')).get_fdata()
    param_gt = all_ground_truth_values[..., full_parameter_lists[model].index(parameter)].flatten()

    mcmc_mean = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                             / model / 'samples' / 'univariate_normal' / f'{parameter}.nii.gz')).get_fdata().flatten()

    return compute_bias(param_gt, mcmc_mean)


def compute_bias_mle_additional_maps(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    ground_truth_value = nib.load(str(data_dir / protocol / model / 'original_parameters_extra_output_maps' /
                                      '2' / f'{parameter}.nii.gz')).get_fdata()

    mle = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                       / model / f'{parameter}.nii.gz')).get_fdata().flatten()

    return compute_bias(ground_truth_value, mle)


def compute_bias_mcmc_additional_maps(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    ground_truth_value = nib.load(str(data_dir / protocol / model / 'original_parameters_extra_output_maps' /
                                      '2' / f'{parameter}.nii.gz')).get_fdata()

    mcmc_mean = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                             / model / 'samples' / 'model_defined_maps' / f'{parameter}.nii.gz')).get_fdata().flatten()

    return compute_bias(ground_truth_value, mcmc_mean)


def compute_bias(param_ground_truth, point_estimate):
    return np.mean(param_ground_truth - point_estimate)


results = []
for protocol in protocols:
    for model in models:
        for snr in noise_snrs:
            for repetition in range(10):
                row = {
                    'protocol': protocol,
                    'model': model,
                    'snr': snr,
                    'repetition': repetition,
                }

                for parameter in randomized_parameter_names[model]:
                    bias_mle = compute_bias_mle(data_dir, protocol, model, snr, repetition, parameter)
                    bias_mcmc = compute_bias_mcmc(data_dir, protocol, model, snr, repetition, parameter)

                    results.append(row | {
                        'parameter': parameter,
                        'method': 'MLE',
                        'bias': bias_mle
                    })

                    results.append(row | {
                        'parameter': parameter,
                        'method': 'MCMC',
                        'bias': bias_mcmc
                    })

                for parameter in additional_parameters.get(model, []):
                    bias_mle = compute_bias_mle_additional_maps(data_dir, protocol, model, snr, repetition, parameter)
                    bias_mcmc = compute_bias_mcmc_additional_maps(data_dir, protocol, model, snr, repetition, parameter)

                    results.append(row | {
                        'parameter': parameter,
                        'method': 'MLE',
                        'bias': bias_mle
                    })

                    results.append(row | {
                        'parameter': parameter,
                        'method': 'MCMC',
                        'bias': bias_mcmc
                    })


pd.DataFrame(results).to_csv('/tmp/overall_bias.csv', index=False)





