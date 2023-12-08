from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

randomized_parameter_names = {
    'NODDI': ['w_ic.w'],
    'BinghamNODDI_r1': ['w_in0.w'],
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


# confidence_z_score = 1.282  # 80%
# confidence_z_score = 1.440 # 85%
# confidence_z_score = 1.645  # 90%
confidence_z_score = 1.96  # 95%

noise_snrs = [
    2,
    5,
    10,
    20,
    30,
    40,
    50]

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


def compute_ecs(param_ground_truth, point_estimate, std_estimate):
    ci_lb = point_estimate - confidence_z_score * std_estimate
    ci_ub = point_estimate + confidence_z_score * std_estimate

    # plt.hist(param_ground_truth)
    # plt.show()

    gt_within_bounds = ((param_ground_truth > ci_lb) & (param_ground_truth < ci_ub)).astype(int)

    return np.sum(gt_within_bounds) / len(param_ground_truth)


results = []
for protocol in protocols:
    for model in models:
        for snr in noise_snrs:
            for repetition in range(10):
                print(protocol, model, snr, repetition)
                for parameter in randomized_parameter_names.get(model, []):
                    ecs_mle = compute_ecs_mle(data_dir, protocol, model, snr, repetition, parameter)
                    ecs_mcmc = compute_ecs_mcmc(data_dir, protocol, model, snr, repetition, parameter)

                    results.append({
                        'protocol': protocol,
                        'model': model,
                        'snr': snr,
                        'repetition': repetition,
                        'parameter': parameter,
                        'method': 'MLE',
                        'ECS': ecs_mle
                    })
                    results.append({
                        'protocol': protocol,
                        'model': model,
                        'snr': snr,
                        'repetition': repetition,
                        'parameter': parameter,
                        'method': 'MCMC',
                        'ECS': ecs_mcmc
                    })

                for parameter in additional_parameters.get(model, []):
                    ecs_mle = compute_ecs_mle_additional_maps(data_dir, protocol, model, snr, repetition, parameter)
                    ecs_mcmc = compute_ecs_mcmc_additional_maps(data_dir, protocol, model, snr, repetition, parameter)

                    results.append({
                        'protocol': protocol,
                        'model': model,
                        'snr': snr,
                        'repetition': repetition,
                        'parameter': parameter,
                        'method': 'MLE',
                        'ECS': ecs_mle
                    })
                    results.append({
                        'protocol': protocol,
                        'model': model,
                        'snr': snr,
                        'repetition': repetition,
                        'parameter': parameter,
                        'method': 'MCMC',
                        'ECS': ecs_mcmc
                    })


pd.DataFrame(results).to_csv('/tmp/overall_ecs.csv', index=False)





