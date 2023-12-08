from itertools import pairwise
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd


randomized_parameter_names = {
    'BallStick_r1': ['w_stick0.w', 'Stick0.theta', 'Stick0.phi'],
    'BallStick_r2': ['w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi'],
    'BallStick_r3': ['w_stick0.w', 'Stick0.theta', 'Stick0.phi', 'w_stick1.w', 'Stick1.theta', 'Stick1.phi',
                              'w_stick2.w', 'Stick2.theta', 'Stick2.phi'],
    'NODDI': ['w_ic.w', 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa', 'w_ec.w'],
    'BinghamNODDI_r1': ['w_in0.w', 'BinghamNODDI_IN0.theta', 'BinghamNODDI_IN0.phi', 'BinghamNODDI_IN0.psi',
                              'BinghamNODDI_IN0.k1', 'BinghamNODDI_IN0.kw', 'w_en0.w'],
    'CHARMED_r1': ['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1',
                              'Tensor.theta', 'Tensor.phi', 'Tensor.psi', 'w_res0.w',
                              'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi'],
    'CHARMED_r2': ['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta', 'Tensor.phi', 'Tensor.psi',
                              'w_res0.w',
                              'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
                              'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi'],
    'CHARMED_r3': ['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1', 'Tensor.theta',
                                        'Tensor.phi', 'Tensor.psi', 'w_res0.w',
                              'CHARMEDRestricted0.d', 'CHARMEDRestricted0.theta', 'CHARMEDRestricted0.phi', 'w_res1.w',
                              'CHARMEDRestricted1.d', 'CHARMEDRestricted1.theta', 'CHARMEDRestricted1.phi', 'w_res2.w',
                              'CHARMEDRestricted2.d', 'CHARMEDRestricted2.theta', 'CHARMEDRestricted2.phi'],
    'Tensor': ['Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1',
                              'Tensor.theta', 'Tensor.phi', 'Tensor.psi'],
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


def compute_precision_mle(data_dir: Path, protocol: str, model: str, snr: int, parameter: str):
    repetitions = []
    for repetition in range(10):
        mle = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                           / model / f'{parameter}.nii.gz')).get_fdata().flatten()
        repetitions.append(mle)

    return compute_precision(repetitions)


def compute_precision_mcmc(data_dir: Path, protocol: str, model: str, snr: int, parameter: str):
    repetitions = []
    for repetition in range(10):
        mcmc_mean = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                             / model / 'samples' / 'univariate_normal' / f'{parameter}.nii.gz')).get_fdata().flatten()
        repetitions.append(mcmc_mean)

    return compute_precision(repetitions)


def compute_precision_mle_additional_maps(data_dir: Path, protocol: str, model: str, snr: int, parameter: str):
    repetitions = []
    for repetition in range(10):
        mle = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                           / model / f'{parameter}.nii.gz')).get_fdata().flatten()
        repetitions.append(mle)

    return compute_precision(repetitions)


def compute_precision_mcmc_additional_maps(data_dir: Path, protocol: str, model: str, snr: int, parameter: str):
    repetitions = []
    for repetition in range(10):
        mcmc_mean = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                                 / model / 'samples' / 'model_defined_maps' / f'{parameter}.nii.gz')).get_fdata().flatten()
        repetitions.append(mcmc_mean)

    return compute_precision(repetitions)


def compute_precision(point_estimates):
    return np.mean(np.abs(point_estimates - np.mean(point_estimates, axis=0)) / len(point_estimates))


results = []
for protocol in protocols:
    for model in models:
        for snr in noise_snrs:
            row = {
                'protocol': protocol,
                'model': model,
                'snr': snr
            }

            for parameter in randomized_parameter_names[model]:
                precision_mle = compute_precision_mle(data_dir, protocol, model, snr, parameter)
                precision_mcmc = compute_precision_mcmc(data_dir, protocol, model, snr, parameter)

                results.append(row | {
                    'parameter': parameter,
                    'method': 'MLE',
                    'precision': precision_mle
                })

                results.append(row | {
                    'parameter': parameter,
                    'method': 'MCMC',
                    'precision': precision_mcmc
                })

            for parameter in additional_parameters.get(model, []):
                precision_mle = compute_precision_mle_additional_maps(data_dir, protocol, model, snr, parameter)
                precision_mcmc = compute_precision_mcmc_additional_maps(data_dir, protocol, model, snr, parameter)

                results.append(row | {
                    'parameter': parameter,
                    'method': 'MLE',
                    'precision': precision_mle
                })

                results.append(row | {
                    'parameter': parameter,
                    'method': 'MCMC',
                    'precision': precision_mcmc
                })


pd.DataFrame(results).to_csv('/tmp/overall_precision.csv', index=False)





