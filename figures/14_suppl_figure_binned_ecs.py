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

display_params = {
    # 'BallStick_r1': ['FS'],
    # 'BallStick_r2': ['FS'],
    # 'BallStick_r3': ['FS'],
    # 'NODDI': ['w_ic.w'],
    # 'BinghamNODDI_r1': ['w_in0.w'],
    'CHARMED_r1': ['FR'],
    # 'Tensor': ['Tensor.FA'],
}

noise_snrs = [2, 5, 10, 20, 30, 40, 50]

def in_display_list(model, parameter):
    return (model in display_params) and (parameter in display_params[model])


output_path = Path('/tmp/uncertainty_paper/suppl_ecs')
output_path.mkdir(exist_ok=True, parents=True)

df = pd.read_csv('/tmp/binned_ecs.csv')
df = df[df[['model', 'parameter']].apply(lambda el: in_display_list(*el), axis=1)]
df = df[df['bin_size'] > 600]


df['model_parameter'] = df['model'] + ' - ' + df['parameter']
df['protocol_method'] = df['protocol'] + ' - ' + df['method']
df['bin_center'] = (df['bin_ub'] + df['bin_lb']) / 2


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
    for snr in noise_snrs:
        fig, axes = plt.subplots(1, 1)
        sns.lineplot(data=df[(df['model_parameter'] == model_parameter) & (df['snr'] == snr)],
                     y='ECS', x='bin_center',
                     hue='protocol_method', ax=axes, palette=colors)
        axes.set_title(model_parameter + ' - SNR' + str(snr))
        axes.set_ylim([axes.get_ylim()[0], 1.05])
        # axes.set_xticks([10, 20, 30, 40, 50])
        plt.show()
        # plt.savefig(str(output_path / f'{model_parameter} - snr {snr}.png'))


# for model_parameter in df['model_parameter'].unique():
#     bin_centers = df[df['model_parameter'] == model_parameter]['bin_center'].unique()
#
#     for bin_ind, bin_center in enumerate(bin_centers):
#         fig, axes = plt.subplots(1, 1)
#         sns.lineplot(data=df[(df['model_parameter'] == model_parameter) & (df['bin_ind'] == bin_ind)],
#                      y='ECS', x='snr',
#                      hue='protocol_method', ax=axes, palette=colors)
#         axes.set_title(model_parameter + ' - bin_center' + str(bin_center))
#         axes.set_ylim([axes.get_ylim()[0], 1.05])
#         plt.show()
#         # plt.savefig(str(output_path / f'{model_parameter} - bin_ind {bin_ind}.png'))
#
