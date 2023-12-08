__author__ = 'Robbert Harms'
__date__ = '2023-09-28'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()

display_params = {
    'BallStick_r1': ['FS'],
    'BallStick_r2': ['FS'],
    'BallStick_r3': ['FS'],
    'NODDI': ['w_ic.w'],
    'BinghamNODDI_r1': ['w_in0.w'],
    'CHARMED_r1': ['FR'],
    'Tensor': ['Tensor.FA'],
}
noise_snrs = [2, 5, 10, 20, 30, 40, 50]


def in_display_list(model, parameter):
    return (model in display_params) and (parameter in display_params[model])


output_path = Path('/tmp/uncertainty_paper/suppl_precision')
output_path.mkdir(exist_ok=True, parents=True)

df = pd.read_csv('/tmp/overall_precision.csv')
df = df[df[['model', 'parameter']].apply(lambda el: in_display_list(*el), axis=1)]

df['model_parameter'] = df['model'] + ' - ' + df['parameter']
df['protocol_method'] = df['protocol'] + ' - ' + df['method']


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)

# set_matplotlib_font_size(22)
colors = ['#c45054', '#0773b3', '#e79f27', '#65e065']
linestyles = ['-', '-', '--', '--']


for model_parameter in df['model_parameter'].unique():
    fig, axes = plt.subplots(1, 1)
    sns.lineplot(data=df[df['model_parameter'] == model_parameter], y='precision', x='snr',
                 hue='protocol_method', ax=axes, palette=colors)
    axes.set_title(model_parameter)
    # axes.set_ylim([axes.get_ylim()[0], 1.05])
    # plt.show()
    plt.savefig(str(output_path / f'{model_parameter}.png'))

# fig, axes = plt.subplots(2, 2)
# sns.lineplot(data=df[df['method'] == 'MLE'], y='precision', x='snr', hue='model_parameter', ax=axes[0][0])
# sns.lineplot(data=df[df['method'] == 'MLE'], y='precision', x='snr', hue='model_parameter', ax=axes[0][1])
# sns.lineplot(data=df[df['method'] == 'MCMC'], y='precision', x='snr', hue='model_parameter', ax=axes[1][0])
# sns.lineplot(data=df[df['method'] == 'MCMC'], y='precision', x='snr', hue='model_parameter', ax=axes[1][1])
#
# plt.show()