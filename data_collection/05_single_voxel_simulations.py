"""Simulation study of a single voxel.

This script first simulates for different models and two different protocols a single voxel
using a single fixed parameter position. From the generated signal we create multiple copies with different SNR.

Before running this script, please copy the protocol files "hcp_mgh_1003.prtcl" and "rheinland_v3a_1_2mm.prtcl"
to the directory where you wish to store the simulation data and results. Set the same path here in this script.
"""

__author__ = 'Robbert Harms'
__date__ = '2020-12-15'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

import os

import mdt
import numpy as np

# the path to where you stored the protocol files.
import mot

data_storage_path = r'/home/robbert/phd-data/papers/uncertainty_paper/single_voxel_simulation/'

# an utility for easily appending to paths
pjoin = mdt.make_path_joiner(data_storage_path)

# The names of the protocol files we wish to simulate data for
protocols = [
    'hcp_mgh_1003',
    'rheinland_v3a_1_2mm'
]

# The models for which we want to simulate the data
models = [
    # 'BallStick_r1',
    # 'BallStick_r2',
    # 'BallStick_r3',
    # 'NODDI',
    'BinghamNODDI_r1',
    # 'Tensor',
    # 'CHARMED_r1',
    # 'CHARMED_r2',
    # 'CHARMED_r3'
]

# The different SNR's we wish to simulate. Each additional SNR will generate a copy of the simulated data with
# noise corresponding to the desired SNR.
noise_snrs = [2, 5, 10, 20, 30, 40, 50]

# The b0 signal intensity, fixed such that we can easily generate the noise (noise_std = unweighted_signal_height / SNR)
unweighted_signal_height = 1e4


# Model specific simulation settings
simulations = {
    'BallStick_r1':
    # S0.s0, w_stick0.w, Stick0.theta, Stick0.phi
        [[unweighted_signal_height, 0.5, np.pi / 2, np.pi / 2]],
    'BallStick_r2':
    # S0.s0, w_stick0.w, Stick0.theta, Stick0.phi, w_stick1.w, Stick1.theta, Stick1.phi
        [[unweighted_signal_height, 0.6, np.pi / 2, np.pi / 2, 0.2, 0.7853981633974483, 0.7853981633974483]],
    'BallStick_r3':
    # S0.s0, w_stick0.w, Stick0.theta, Stick0.phi, w_stick1.w, Stick1.theta, Stick1.phi, w_stick2.w, Stick2.theta, Stick2.phi
        [[unweighted_signal_height, 0.4, np.pi / 2, np.pi / 2, 0.2, 0.7853981633974483, 0.7853981633974483, 0.2,
          2.356194490192345, 2.356194490192345]],
    'NODDI':
    # S0.s0, w_ic.w, NODDI_IC.theta, NODDI_IC.phi, NODDI_IC.kappa, w_ec.w
        [[unweighted_signal_height, 0.5, np.pi / 2, np.pi / 2, 10, 0.3]],
    'BinghamNODDI_r1':
    # S0.s0, w_in0.w, BinghamNODDI_IN0.theta, BinghamNODDI_IN0.phi, BinghamNODDI_IN0.kappa, BinghamNODDI_IN0.k1, BinghamNODDI_IN0.kw, w_en0.w
        [[unweighted_signal_height, 0.5, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 10, 10, 0.3]],
    'CHARMED_r1':
    # S0.s0, Tensor.d, Tensor.dperp0, Tensor.dperp1, Tensor.theta, Tensor.phi, Tensor.psi, w_res0.w, CHARMEDRestricted0.d, CHARMEDRestricted0.theta, CHARMEDRestricted0.phi
        [[unweighted_signal_height, 5e-10, 2.5e-10, 2.5e-10, np.pi / 2, np.pi / 2, np.pi / 2, 0.5, 5e-10,
          np.pi / 2, np.pi / 2]],
    'CHARMED_r2':
    # S0.s0, Tensor.d, Tensor.dperp0, Tensor.dperp1, Tensor.theta, Tensor.phi, Tensor.psi, w_res0.w, CHARMEDRestricted0.d, CHARMEDRestricted0.theta, CHARMEDRestricted0.phi, w_res1.w, CHARMEDRestricted1.d, CHARMEDRestricted1.theta, CHARMEDRestricted1.phi
        [[unweighted_signal_height, 5e-10, 2.5e-10, 5e-11, np.pi / 2, np.pi / 2, np.pi / 2, 0.3, 5e-10,
          np.pi / 2, np.pi / 2, 0.3, 1e-10, 0.7853981633974483, 0.7853981633974483]],
    'CHARMED_r3':
    # S0.s0, Tensor.d, Tensor.dperp0, Tensor.dperp1, Tensor.theta, Tensor.phi, Tensor.psi, w_res0.w, CHARMEDRestricted0.d, CHARMEDRestricted0.theta, CHARMEDRestricted0.phi, w_res1.w, CHARMEDRestricted1.d, CHARMEDRestricted1.theta, CHARMEDRestricted1.phi, w_res2.w, CHARMEDRestricted2.d, CHARMEDRestricted2.theta, CHARMEDRestricted2.phi
        [[unweighted_signal_height, 5e-10, 2.5e-10, 5e-11, np.pi / 2, np.pi / 2, np.pi / 2, 0.2, 5e-10,
          np.pi / 2, np.pi / 2, 0.2, 1e-10, 0.7853981633974483, 0.7853981633974483, 0.2, 5e-11,
          2.356194490192345, 2.356194490192345]],
    'Tensor':
    # S0.s0, Tensor.d, Tensor.dperp0, Tensor.dperp1, Tensor.theta, Tensor.phi, Tensor.psi
        [[unweighted_signal_height, 5e-10, 2.5e-10, 5e-11, np.pi / 2, np.pi / 2, np.pi / 2]]
}


def create_simulations(protocol_name, model_name):
    """Create a simulation cube for a single parameter combination (single voxel)."""
    # print(mdt.get_cl_devices())
    mot.configuration.set_cl_environments(1)

    output_pjoin = pjoin.create_extended(protocol_name, model_name)
    if os.path.exists(output_pjoin()):
        return
    else:
        output_pjoin.make_dirs()

    model = mdt.get_model(model_name)(volume_selection=False)

    parameters = np.array(simulations[model_name])

    simulated_signals = mdt.simulate_signals(model, mdt.load_protocol(pjoin(protocol_name)), parameters)

    mdt.write_nifti(parameters[None, :, None, :], output_pjoin('original_parameters.nii'))
    mdt.write_nifti(simulated_signals[None, :, None, :], output_pjoin('simulated_signals.nii'))

    for snr in noise_snrs:
        noisy_signals = mdt.add_rician_noise(simulated_signals, unweighted_signal_height / snr, seed=0)
        mdt.write_nifti(noisy_signals[None, :, None, :], output_pjoin('noisy_signals_{}.nii'.format(snr)))

    mdt.create_blank_mask(output_pjoin('noisy_signals_{}.nii'.format(noise_snrs[0])),
                          output_pjoin('mask.nii'))


for protocol_name in protocols:
    for model_name in models:
        print(protocol_name, model_name)
        create_simulations(protocol_name, model_name)

