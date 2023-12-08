from itertools import pairwise
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


randomized_parameter_names = {
    'BallStick_r1': ['w_stick0.w',
                     # 'Stick0.theta', 'Stick0.phi'
                     ],
    'BallStick_r2': ['w_stick0.w',
                     # 'Stick0.theta', 'Stick0.phi',
                     'w_stick1.w',
                     # 'Stick1.theta', 'Stick1.phi'
                     ],
    'BallStick_r3': ['w_stick0.w',
                     # 'Stick0.theta', 'Stick0.phi',
                     'w_stick1.w',
                     # 'Stick1.theta', 'Stick1.phi',
                    'w_stick2.w',
                     # 'Stick2.theta', 'Stick2.phi'
                     ],
    'NODDI': ['w_ic.w',
              # 'NODDI_IC.theta', 'NODDI_IC.phi', 'NODDI_IC.kappa',
              'w_ec.w'],
    'BinghamNODDI_r1': ['w_in0.w',
                        # 'BinghamNODDI_IN0.theta', 'BinghamNODDI_IN0.phi', 'BinghamNODDI_IN0.psi',
                        #       'BinghamNODDI_IN0.k1', 'BinghamNODDI_IN0.kw', 'w_en0.w'
                        ],
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
    'Tensor': [
        # 'Tensor.d', 'Tensor.dperp0', 'Tensor.dperp1',
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
    # 'BallStick_r1',
    # 'BallStick_r2',
    # 'BallStick_r3',
    # 'NODDI',
    # 'BinghamNODDI_r1',
    'Tensor',
    # 'CHARMED_r1',
]

# confidence_z_score = 1.440 # 85
# confidence_z_score = 1.645  # 90%
confidence_z_score = 1.96  # 95%

noise_snrs = [2, 5, 10, 20, 30, 40, 50]

data_dir = Path('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/')


def compute_ecs_mle(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    all_ground_truth_values = nib.load(str(data_dir / protocol / model / 'original_parameters.nii')).get_fdata()
    param_gt = all_ground_truth_values[..., full_parameter_lists[model].index(parameter)].flatten()

    mle = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                       / model / f'{parameter}.nii.gz')).get_fdata().flatten()
    mle_std = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                           / model / f'{parameter}.std.nii.gz')).get_fdata().flatten()

    return compute_ecs(param_gt, mle, mle_std)


def compute_ecs_mcmc(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    all_ground_truth_values = nib.load(str(data_dir / protocol / model / 'original_parameters.nii')).get_fdata()
    param_gt = all_ground_truth_values[..., full_parameter_lists[model].index(parameter)].flatten()

    mcmc_mean = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                             / model / 'samples' / 'univariate_normal' / f'{parameter}.nii.gz')).get_fdata().flatten()
    mcmc_std = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                            / model / 'samples' / 'univariate_normal' /
                            f'{parameter}.std.nii.gz')).get_fdata().flatten()

    return compute_ecs(param_gt, mcmc_mean, mcmc_std)


def compute_ecs_mle_additional_maps(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    ground_truth_value = nib.load(str(data_dir / protocol / model / 'original_parameters_extra_output_maps' /
                                      '2' / f'{parameter}.nii.gz')).get_fdata()

    mle = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                       / model / f'{parameter}.nii.gz')).get_fdata().flatten()
    mle_std = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                           / model / f'{parameter}.std.nii.gz')).get_fdata().flatten()

    if model == 'CHARMED_r1' and parameter == 'FR':
        lb = 0.2
        ub = 0.8
        return compute_ecs(ground_truth_value, mle, mle_std, lb=lb, ub=ub)

    return compute_ecs(ground_truth_value, mle, mle_std)


def compute_ecs_mcmc_additional_maps(data_dir: Path, protocol: str, model: str, snr: int, repetition: int, parameter: str):
    ground_truth_value = nib.load(str(data_dir / protocol / model / 'original_parameters_extra_output_maps' /
                                      '2' / f'{parameter}.nii.gz')).get_fdata()

    mcmc_mean = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                             / model / 'samples' / 'model_defined_maps' / f'{parameter}.nii.gz')).get_fdata().flatten()
    mcmc_std = nib.load(str(data_dir / protocol / model / 'output' / str(snr) / str(repetition)
                            / model / 'samples' / 'model_defined_maps' /
                            f'{parameter}.std.nii.gz')).get_fdata().flatten()

    return compute_ecs(ground_truth_value, mcmc_mean, mcmc_std)


def compute_ecs(param_ground_truth, point_estimate, std_estimate, lb=None, ub=None):
    ci_lb = point_estimate - confidence_z_score * std_estimate
    ci_ub = point_estimate + confidence_z_score * std_estimate

    gt_within_bounds = ((param_ground_truth > ci_lb) & (param_ground_truth < ci_ub)).astype(int)

    if lb is None:
        minimum = np.min(param_ground_truth).round(2)
    else:
        minimum = lb
    if ub is None:
        maximum = np.max(param_ground_truth).round(2)
    else:
        maximum = ub

    bin_edges = np.linspace(minimum, maximum, 11)

    plt.hist(std_estimate, bins=100)
    plt.show()

    ecs = []
    for bin_ind, (bin_lb, bin_ub) in enumerate(pairwise(bin_edges)):
        trials_in_bin = gt_within_bounds[(param_ground_truth >= bin_lb) & (param_ground_truth < bin_ub)]
        ecs.append({
            'bin_lb': bin_lb,
            'bin_ub': bin_ub,
            'bin_ind': bin_ind,
            'ECS': np.sum(trials_in_bin) / len(trials_in_bin),
            'bin_size': len(trials_in_bin)
        })
    return ecs


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
                    ecs_mle = compute_ecs_mle(data_dir, protocol, model, snr, repetition, parameter)
                    ecs_mcmc = compute_ecs_mcmc(data_dir, protocol, model, snr, repetition, parameter)

                    for ecs_result in ecs_mle:
                        results.append(row | {
                            'parameter': parameter,
                            'method': 'MLE'
                        } | ecs_result)

                    for ecs_result in ecs_mcmc:
                        results.append(row | {
                            'parameter': parameter,
                            'method': 'MCMC'
                        } | ecs_result)

                for parameter in additional_parameters.get(model, []):
                    ecs_mle = compute_ecs_mle_additional_maps(data_dir, protocol, model, snr, repetition, parameter)
                    ecs_mcmc = compute_ecs_mcmc_additional_maps(data_dir, protocol, model, snr, repetition, parameter)

                    for ecs_result in ecs_mle:
                        results.append(row | {
                            'parameter': parameter,
                            'method': 'MLE'
                        } | ecs_result)

                    for ecs_result in ecs_mcmc:
                        results.append(row | {
                            'parameter': parameter,
                            'method': 'MCMC'
                        } | ecs_result)



pd.DataFrame(results).to_csv('/tmp/binned_ecs.csv', index=False)





