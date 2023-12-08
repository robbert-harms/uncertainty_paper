__author__ = 'Robbert Harms'
__date__ = '2023-09-28'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()

import subprocess

from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator
import numpy as np


display_params = {
    'BallStick_r1': ['FS'],
    # 'BallStick_r2': ['FS'],
    # 'BallStick_r3': ['FS'],
    'NODDI': ['w_ic.w'],
    # 'BinghamNODDI_r1': ['FR'],
    'CHARMED_r1': ['FR'],
    'Tensor': ['Tensor.FA'],
}
noise_snrs = [2, 5, 10, 20, 30, 40, 50]


def in_display_list(model, parameter):
    return (model in display_params) and (parameter in display_params[model])


output_path = Path('/tmp/uncertainty_paper/suppl_ecs')
output_path.mkdir(exist_ok=True, parents=True)

df = pd.read_csv('/tmp/overall_ecs.csv')
df = df[df[['model', 'parameter']].apply(lambda el: in_display_list(*el), axis=1)]


df['model_parameter'] = df['model'] + ' - ' + df['parameter']
df['protocol_method'] = df['protocol'] + ' - ' + df['method']
df = df[df['snr'] > 5]
df = df[df['protocol'] == 'rheinland_v3a_1_2mm']


def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)

set_matplotlib_font_size(22)
colors = [
    # '#c45054', '#0773b3',
    '#e79f27', '#65e065']
# linestyles = ['-', '-', '--', '--']
# linestyles = [(1, 0), (1, 0), (2, 2), (2, 2)]
linestyles = [(2, 2), (2, 2)]

titles = {
    'Tensor - Tensor.FA': 'Tensor - FA',
    'BallStick_r1 - FS': 'BallStick_in1 - FS',
    'NODDI - w_ic.w': 'NODDI - FR',
    'CHARMED_r1 - FR': 'CHARMED_in1 - FR',
    'BinghamNODDI_r1 - FR': 'BinghamNODDI_in1 - FR',
}

for model_parameter in df['model_parameter'].unique():
    fig, axis = plt.subplots(1, 1)
    sns.lineplot(data=df[df['model_parameter'] == model_parameter], y='ECS', x='snr',
                 hue='protocol_method', style='protocol_method', ax=axis,
                 palette=colors,
                 dashes=linestyles,
                 # linewidth=4,
                 )

    axis.set_title(titles[model_parameter])
    # axes.set_ylim([axes.get_ylim()[0], 1.05])
    axis.set_ylim([0.8, 1.05])
    axis.set_xticks(noise_snrs[2:])
    axis.get_legend().remove()
    axis.set_xlabel('SNR (a.u.)')
    axis.set_ylabel('ECS (a.u.)')

    # plt.show()
    plt.gcf().subplots_adjust(bottom=0.18, top=0.9, left=0.20, right=0.94)
    plt.savefig(str(output_path / f'{model_parameter}.png'))

# fig, axes = plt.subplots(2, 2)
# sns.lineplot(data=df[df['method'] == 'MLE'], y='ECS', x='snr', hue='model_parameter', ax=axes[0][0])
# sns.lineplot(data=df[df['method'] == 'MLE'], y='ECS', x='snr', hue='model_parameter', ax=axes[0][1])
# sns.lineplot(data=df[df['method'] == 'MCMC'], y='ECS', x='snr', hue='model_parameter', ax=axes[1][0])
# sns.lineplot(data=df[df['method'] == 'MCMC'], y='ECS', x='snr', hue='model_parameter', ax=axes[1][1])
#
# plt.show()