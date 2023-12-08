__author__ = 'Robbert Harms'
__date__ = '2023-09-28'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

from pathlib import Path
import nibabel as nib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


data_dir = Path('/home/robbert/phd-data/papers/uncertainty_paper/snr_simulations/')

model = 'BallStick_r1'
protocol = 'hcp_mgh_1003'
param_name = 'w_stick0.w'
param_ind = 1
snr = 5
repetition = 0


ground_truth = nib.load(str(data_dir / protocol / model / 'original_parameters.nii')).get_fdata().flatten()
all_ground_truth_values = nib.load(str(data_dir / protocol / model / 'original_parameters.nii')).get_fdata()
param_gt = all_ground_truth_values[..., param_ind].flatten()

crlb = nib.load(str(data_dir / protocol / model / 'CRLB'
                    / str(snr) / model / 'FIM' / f'{param_name}.std.nii.gz')).get_fdata().flatten()

mle = nib.load(str(data_dir / protocol / model / 'output' /
                       str(snr) / str(repetition) / model / f'{param_name}.nii.gz')).get_fdata().flatten()
mle_fim = nib.load(str(data_dir / protocol / model / 'output' /
                       str(snr) / str(repetition) / model / f'{param_name}.std.nii.gz')).get_fdata().flatten()


mcmc_std = nib.load(str(data_dir / protocol / model / 'output' /
                        str(snr) / str(repetition) / model / 'samples' / 'univariate_normal' /
                        f'{param_name}.std.nii.gz')).get_fdata().flatten()


crlb_mle = (1 + (mle - param_gt))**2 / crlb


fig, axes = plt.subplots(1, 1)
axes.plot(crlb_mle, label='CRLB')
axes.plot(mle_fim, label='MLE')
# axes.plot(mcmc_std, label='MCMC')
axes.legend()

# df = pd.DataFrame({'crlb': crlb, 'mle_fim': mle_fim, 'mcmc_std': mcmc_std})
# sns.lineplot(data=df)
plt.show()


print()


