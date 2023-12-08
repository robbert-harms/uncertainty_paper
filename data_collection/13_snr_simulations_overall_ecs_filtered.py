__author__ = 'Robbert Harms'
__date__ = '2023-10-05'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

protocols = [
    'hcp_mgh_1003',
    'rheinland_v3a_1_2mm'
]

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

diffusivity_boundary_cutoff = 0.5e-8


def ballstick_r1_fs_ecs(protocol):
    fs_ground_truth = nib.load(str(data_dir / protocol / 'BallStick_r1' / 'original_parameters_extra_output_maps' /
                                   '50' / f'FS.nii.gz')).get_fdata().flatten()

    results = []
    for snr in noise_snrs:
        for repetition in range(10):
            fs_mle_est = nib.load(str(data_dir / protocol / 'BallStick_r1' / 'output' /
                                  str(snr) / str(repetition) / 'BallStick_r1' / 'FS.nii.gz')).get_fdata().flatten()
            fs_mle_std_est = nib.load(str(data_dir / protocol / 'BallStick_r1' / 'output' /
                                  str(snr) / str(repetition) / 'BallStick_r1' / 'FS.std.nii.gz')).get_fdata().flatten()

            fs_mcmc_est = nib.load(str(data_dir / protocol / 'BallStick_r1' / 'output' /
                                       str(snr) / str(repetition) / 'BallStick_r1' /
                                       'samples' / 'model_defined_maps' / 'FS.nii.gz')).get_fdata().flatten()
            fs_mcmc_std_est = nib.load(str(data_dir / protocol / 'BallStick_r1' / 'output' /
                                           str(snr) / str(repetition) / 'BallStick_r1' /
                                       'samples' / 'model_defined_maps' / 'FS.std.nii.gz')).get_fdata().flatten()

            # locations = (fs_mle_est >= 0.0)
            # plt.scatter(fs_ground_truth[locations], fs_mle_est[locations])
            # plt.show()

            ecs = compute_ecs(fs_ground_truth, fs_mle_est, fs_mle_std_est)
            results.append({
                'model': 'BallStick_r1',
                'parameter': 'FS',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MLE'
            })

            ecs = compute_ecs(fs_ground_truth, fs_mcmc_est, fs_mcmc_std_est)
            results.append({
                'model': 'BallStick_r1',
                'parameter': 'FS',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MCMC'
            })
    return results


def ballstick_r2_fs_ecs(protocol):
    fs_ground_truth = nib.load(str(data_dir / protocol / 'BallStick_r2' / 'original_parameters_extra_output_maps' /
                                   '50' / f'FS.nii.gz')).get_fdata().flatten()

    results = []
    for snr in noise_snrs:
        for repetition in range(10):
            fs_mle_est = nib.load(str(data_dir / protocol / 'BallStick_r2' / 'output' /
                                  str(snr) / str(repetition) / 'BallStick_r2' / 'FS.nii.gz')).get_fdata().flatten()
            fs_mle_std_est = nib.load(str(data_dir / protocol / 'BallStick_r2' / 'output' /
                                  str(snr) / str(repetition) / 'BallStick_r2' / 'FS.std.nii.gz')).get_fdata().flatten()

            fs_mcmc_est = nib.load(str(data_dir / protocol / 'BallStick_r2' / 'output' /
                                       str(snr) / str(repetition) / 'BallStick_r2' /
                                       'samples' / 'model_defined_maps' / 'FS.nii.gz')).get_fdata().flatten()
            fs_mcmc_std_est = nib.load(str(data_dir / protocol / 'BallStick_r2' / 'output' /
                                           str(snr) / str(repetition) / 'BallStick_r2' /
                                       'samples' / 'model_defined_maps' / 'FS.std.nii.gz')).get_fdata().flatten()

            locations = (fs_mle_est >= 0.1)
            # plt.scatter(fs_ground_truth[locations], fs_mle_est[locations])
            # plt.show()

            ecs = compute_ecs(fs_ground_truth[locations], fs_mle_est[locations], fs_mle_std_est[locations])
            results.append({
                'model': 'BallStick_r2',
                'parameter': 'FS',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MLE'
            })

            ecs = compute_ecs(fs_ground_truth[locations], fs_mcmc_est[locations], fs_mcmc_std_est[locations])
            results.append({
                'model': 'BallStick_r2',
                'parameter': 'FS',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MCMC'
            })
    return results


def tensor_fa_ecs(protocol):
    fa_ground_truth = nib.load(str(data_dir / protocol / 'Tensor' / 'original_parameters_extra_output_maps' /
                                   '50' / f'Tensor.FA.nii.gz')).get_fdata().flatten()

    results = []
    for snr in noise_snrs:
        for repetition in range(10):
            fa_mle_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                  str(snr) / str(repetition) / 'Tensor' / 'Tensor.FA.nii.gz')).get_fdata().flatten()
            fa_mle_std_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                  str(snr) / str(repetition) / 'Tensor' / 'Tensor.FA.std.nii.gz')).get_fdata().flatten()

            fa_mcmc_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                       str(snr) / str(repetition) / 'Tensor' /
                                       'samples' / 'model_defined_maps' / 'Tensor.FA.nii.gz')).get_fdata().flatten()
            fa_mcmc_std_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                           str(snr) / str(repetition) / 'Tensor' /
                                       'samples' / 'model_defined_maps' / 'Tensor.FA.std.nii.gz')).get_fdata().flatten()

            d_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                  str(snr) / str(repetition) / 'Tensor' / 'Tensor.d.nii.gz')).get_fdata().flatten()
            d_perp0_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                 str(snr) / str(repetition) / 'Tensor' / 'Tensor.dperp0.nii.gz')).get_fdata().flatten()
            d_perp1_est = nib.load(str(data_dir / protocol / 'Tensor' / 'output' /
                                       str(snr) / str(repetition) / 'Tensor' / 'Tensor.dperp1.nii.gz')).get_fdata().flatten()

            locations = ((d_est <= diffusivity_boundary_cutoff)
                         & (d_perp0_est <= diffusivity_boundary_cutoff)
                         & (d_perp1_est <= diffusivity_boundary_cutoff))

            # plt.scatter(fa_ground_truth[locations], fa_mle_est[locations])
            # plt.show()

            ecs = compute_ecs(fa_ground_truth[locations], fa_mle_est[locations], fa_mle_std_est[locations])
            results.append({
                'model': 'Tensor',
                'parameter': 'Tensor.FA',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MLE'
            })

            ecs = compute_ecs(fa_ground_truth[locations], fa_mcmc_est[locations], fa_mcmc_std_est[locations])
            results.append({
                'model': 'Tensor',
                'parameter': 'Tensor.FA',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MCMC'
            })
    return results


def noddi_fr_ecs(protocol):
    fr_ground_truth = nib.load(str(data_dir / protocol / 'NODDI' / 'original_parameters_extra_output_maps' /
                                   '50' / f'w_ic.w.nii.gz')).get_fdata().flatten()

    results = []
    for snr in noise_snrs:
        for repetition in range(10):
            fr_mle_est = nib.load(str(data_dir / protocol / 'NODDI' / 'output' /
                                  str(snr) / str(repetition) / 'NODDI' / 'w_ic.w.nii.gz')).get_fdata().flatten()
            fr_mle_std_est = nib.load(str(data_dir / protocol / 'NODDI' / 'output' /
                                  str(snr) / str(repetition) / 'NODDI' / 'w_ic.w.std.nii.gz')).get_fdata().flatten()

            fr_mcmc_est = nib.load(str(data_dir / protocol / 'NODDI' / 'output' /
                                       str(snr) / str(repetition) / 'NODDI' /
                                       'samples' / 'univariate_normal' / 'w_ic.w.nii.gz')).get_fdata().flatten()
            fr_mcmc_std_est = nib.load(str(data_dir / protocol / 'NODDI' / 'output' /
                                           str(snr) / str(repetition) / 'NODDI' /
                                       'samples' / 'univariate_normal' / 'w_ic.w.std.nii.gz')).get_fdata().flatten()

            # locations = (fr_mcmc_est >= 0.05)
            # plt.hist(kappa_est)
            # plt.scatter(fr_mle_std_est, fr_mcmc_std_est)
            # plt.show()

            ecs = compute_ecs(fr_ground_truth, fr_mle_est, fr_mle_std_est)
            results.append({
                'model': 'NODDI',
                'parameter': 'w_ic.w',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MLE'
            })

            ecs = compute_ecs(fr_ground_truth, fr_mcmc_est, fr_mcmc_std_est)
            results.append({
                'model': 'NODDI',
                'parameter': 'w_ic.w',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MCMC'
            })
    return results


# def bingham_noddi_fr_ecs(protocol):
#     # w_in0_w_ground_truth = nib.load(str(data_dir / protocol / 'BinghamNODDI_r1' / f'original_parameters.nii')
#     #                              ).get_fdata()[0, :, 0, 1].flatten()
#     w_en0_w_ground_truth = nib.load(str(data_dir / protocol / 'BinghamNODDI_r1' / f'original_parameters.nii')
#                                  ).get_fdata()[0, :, 0, 7].flatten()
#
#     results = []
#     for snr in noise_snrs:
#         for repetition in range(10):
#             fr_mle_est = nib.load(str(data_dir / protocol / 'BinghamNODDI_r1' / 'output' /
#                                   str(snr) / str(repetition) / 'BinghamNODDI_r1' / 'w_en0.w.nii.gz')).get_fdata().flatten()
#             fr_mle_std_est = nib.load(str(data_dir / protocol / 'BinghamNODDI_r1' / 'output' /
#                                   str(snr) / str(repetition) / 'BinghamNODDI_r1' / 'w_en0.w.std.nii.gz')).get_fdata().flatten()
#
#             fr_mcmc_est = nib.load(str(data_dir / protocol / 'BinghamNODDI_r1' / 'output' /
#                                        str(snr) / str(repetition) / 'BinghamNODDI_r1' /
#                                        'samples' / 'univariate_normal' / 'w_en0.w.nii.gz')).get_fdata().flatten()
#             fr_mcmc_std_est = nib.load(str(data_dir / protocol / 'BinghamNODDI_r1' / 'output' /
#                                            str(snr) / str(repetition) / 'BinghamNODDI_r1' /
#                                        'samples' / 'univariate_normal' / 'w_en0.w.std.nii.gz')).get_fdata().flatten()
#
#             # locations = (fr_mcmc_est >= 0.05)
#             # plt.hist(kappa_est)
#             # plt.scatter(fr_mle_std_est, fr_mcmc_std_est)
#             # plt.show()
#
#             ecs = compute_ecs(w_en0_w_ground_truth, fr_mle_est, fr_mle_std_est)
#             results.append({
#                 'model': 'BinghamNODDI_r1',
#                 'parameter': 'FR',
#                 'snr': snr,
#                 'repetition': repetition,
#                 'ECS': ecs,
#                 'method': 'MLE'
#             })
#
#             ecs = compute_ecs(w_en0_w_ground_truth, fr_mcmc_est, fr_mcmc_std_est)
#             results.append({
#                 'model': 'BinghamNODDI_r1',
#                 'parameter': 'FR',
#                 'snr': snr,
#                 'repetition': repetition,
#                 'ECS': ecs,
#                 'method': 'MCMC'
#             })
#     return results


def charmed_fr_ecs(protocol):
    fr_ground_truth = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'original_parameters_extra_output_maps' /
                                   '50' / f'FR.nii.gz')).get_fdata().flatten()

    results = []
    for snr in noise_snrs:
        for repetition in range(10):
            fr_mle_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                  str(snr) / str(repetition) / 'CHARMED_r1' / 'FR.nii.gz')).get_fdata().flatten()
            fr_mle_std_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                  str(snr) / str(repetition) / 'CHARMED_r1' / 'FR.std.nii.gz')).get_fdata().flatten()

            fr_mcmc_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                       str(snr) / str(repetition) / 'CHARMED_r1' /
                                       'samples' / 'model_defined_maps' / 'FR.nii.gz')).get_fdata().flatten()
            fr_mcmc_std_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                           str(snr) / str(repetition) / 'CHARMED_r1' /
                                       'samples' / 'model_defined_maps' / 'FR.std.nii.gz')).get_fdata().flatten()

            res_d_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                 str(snr) / str(repetition) / 'CHARMED_r1'
                                     / 'CHARMEDRestricted0.d.nii.gz')).get_fdata().flatten()
            tensor_d_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                 str(snr) / str(repetition) / 'CHARMED_r1' / 'Tensor.d.nii.gz')).get_fdata().flatten()
            tensor_d_perp0_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                        str(snr) / str(repetition) / 'CHARMED_r1' / 'Tensor.dperp0.nii.gz')).get_fdata().flatten()
            tensor_d_perp1_est = nib.load(str(data_dir / protocol / 'CHARMED_r1' / 'output' /
                                             str(snr) / str(
                repetition) / 'CHARMED_r1' / 'Tensor.dperp1.nii.gz')).get_fdata().flatten()

            locations = ((tensor_d_est <= diffusivity_boundary_cutoff)
                         & (tensor_d_perp0_est <= diffusivity_boundary_cutoff)
                         & (tensor_d_perp1_est <= diffusivity_boundary_cutoff))

            # locations = (fr_mcmc_est >= 0.05)
            # plt.scatter(fr_ground_truth[locations], fr_mle_est[locations])
            # plt.show()

            ecs = compute_ecs(fr_ground_truth[locations], fr_mle_est[locations], fr_mle_std_est[locations])
            results.append({
                'model': 'CHARMED_r1',
                'parameter': 'FR',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MLE'
            })

            ecs = compute_ecs(fr_ground_truth[locations], fr_mcmc_est[locations], fr_mcmc_std_est[locations])
            results.append({
                'model': 'CHARMED_r1',
                'parameter': 'FR',
                'snr': snr,
                'repetition': repetition,
                'ECS': ecs,
                'method': 'MCMC'
            })
    return results


def compute_ecs(param_ground_truth, point_estimate, std_estimate):
    ci_lb = point_estimate - confidence_z_score * std_estimate
    ci_ub = point_estimate + confidence_z_score * std_estimate

    # plt.hist(param_ground_truth)
    # plt.show()

    gt_within_bounds = ((param_ground_truth > ci_lb) & (param_ground_truth < ci_ub)).astype(int)

    return np.sum(gt_within_bounds) / len(param_ground_truth)


results = []
for protocol in protocols:
    tensor_results = tensor_fa_ecs(protocol)
    for tensor_result in tensor_results:
        results.append({'protocol': protocol} | tensor_result)

    ballstick_r1_fs_results = ballstick_r1_fs_ecs(protocol)
    for ballstick_r1_fs_result in ballstick_r1_fs_results:
        results.append({'protocol': protocol} | ballstick_r1_fs_result)

    ballstick_r2_fs_results = ballstick_r2_fs_ecs(protocol)
    for ballstick_r2_fs_result in ballstick_r2_fs_results:
        results.append({'protocol': protocol} | ballstick_r2_fs_result)

    noddi_fr_results = noddi_fr_ecs(protocol)
    for noddi_fr_result in noddi_fr_results:
        results.append({'protocol': protocol} | noddi_fr_result)

    # bingham_noddi_fr_results = bingham_noddi_fr_ecs(protocol)
    # for bingham_noddi_fr_result in bingham_noddi_fr_results:
    #     results.append({'protocol': protocol} | bingham_noddi_fr_result)

    charmed_fr_results = charmed_fr_ecs(protocol)
    for charmed_fr_result in charmed_fr_results:
        results.append({'protocol': protocol} | charmed_fr_result)


pd.DataFrame(results).to_csv('/tmp/overall_ecs.csv', index=False)