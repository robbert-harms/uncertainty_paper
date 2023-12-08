__author__ = 'Robbert Harms'
__date__ = '2023-10-04'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

import nibabel as nib
import matplotlib.pyplot as plt

true_fa = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/original_parameters_extra_output_maps/20/Tensor.FA.nii.gz').get_fdata()
true_d = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/original_parameters.nii').get_fdata()[..., 1]

fa = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/output/20/0/Tensor/Tensor.FA.nii.gz').get_fdata()
fa_std = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/output/20/0/Tensor/Tensor.FA.std.nii.gz').get_fdata()

theta_std = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/output/20/0/Tensor/Tensor.theta.std.nii.gz').get_fdata()
d = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/output/20/0/Tensor/Tensor.dperp0.nii.gz').get_fdata()
d_std = nib.load('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/hcp_mgh_1003/Tensor/output/20/0/Tensor/Tensor.d.std.nii.gz').get_fdata()

plt.scatter(true_fa.flatten(), fa.flatten(), c=d_std.flatten())
plt.colorbar()
plt.show()


print()