__author__ = 'Robbert Harms'
__date__ = '2022-02-26'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert@xkls.nl'
__licence__ = 'LGPL v3'

import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numdifftools import Hessian
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm, truncnorm
import matplotlib.gridspec as gridspec

sns.set()

nmr_samples = 10000
x_range = np.linspace(-5, 5, nmr_samples)
x_min = np.min(x_range)
x_max = np.max(x_range)

trunc_loc, trunc_scale = 0, 1
trunc_a, trunc_b = (0 - trunc_loc) / trunc_scale, (5 - trunc_loc) / trunc_scale

distributions = {
    'simple_norm': norm.rvs(loc=0, scale=1, size=nmr_samples, random_state=0),
    'multimodal_norm': np.hstack([norm.rvs(loc=0, scale=1, size=nmr_samples // 8 * 5, random_state=0),
                                    norm.rvs(loc=3, scale=0.8, size=nmr_samples//8 * 3, random_state=0)]),
    'skewnorm': skewnorm.rvs(10, scale=2, size=nmr_samples, random_state=0),
    'truncnorm': truncnorm.rvs(trunc_a, trunc_b, loc=trunc_loc, scale=trunc_scale, size=nmr_samples, random_state=0),
    'modulus': norm.rvs(loc=0, scale=0.5, size=nmr_samples, random_state=0) % np.pi,
}

titles = {
    'simple_norm': 'Normal',
    'multimodal_norm': 'Multimodal normal',
    'skewnorm': 'Skewed normal',
    'truncnorm': 'Truncated normal',
    'modulus': 'Normal mod $\pi$'
}

models = {
    'simple_norm': lambda x: -norm.logpdf(x, loc=0, scale=1),
    'multimodal_norm': lambda x: -np.log(5/8 * norm.pdf(x, loc=0, scale=1) + 3/8 * norm.pdf(x, loc=3, scale=0.8)),
    'skewnorm': lambda x: -skewnorm.logpdf(x, 10, scale=2),
    'truncnorm': lambda x: -truncnorm.logpdf(x, trunc_a, trunc_b, loc=trunc_loc, scale=trunc_scale),
    'modulus': lambda x: -norm.logpdf(x, loc=0, scale=np.sqrt(0.5)),
}

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(3, 2)
axes = [
    fig.add_subplot(gs[0, :]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
]

letters = itertools.cycle(['A', 'B', 'C', 'D', 'E'])

for ind, (key, samples) in enumerate(distributions.items()):
    estimated_mean = np.mean(samples)
    estimated_std = np.std(samples)
    estimated_mean_distr = norm.pdf(x_range, loc=estimated_mean, scale=estimated_std)

    optimization_results = minimize(models[key], np.array([0]), method='Nelder-Mead', bounds=[(-5, 5)])
    estimated_mode = optimization_results.x[0]
    hessian = Hessian(models[key], step=0.001, method='forward')(estimated_mode)
    estimated_fim = 1 / hessian[0, 0]
    estimated_mode_distr = norm.pdf(x_range, loc=estimated_mode, scale=estimated_fim)

    ax = axes[ind]
    n, _, _ = ax.hist(samples, bins=30, density=True, color='gray', edgecolor='black', linewidth=0.8)
    mode_scale_factor = np.max(n) / np.max(estimated_mode_distr)

    ax.plot(x_range, estimated_mean_distr, c='#56b4e9', linewidth=3)
    ax.plot(x_range, estimated_mode_distr * mode_scale_factor, c='#d55e00', linewidth=3)
    ax.set_title(titles[key])

    if ind == 0:
        ax.annotate(next(letters) + ')', (-0.048, 1.1), xycoords='axes fraction', weight='bold')
    else:
        ax.annotate(next(letters) + ')', (-0.1, 1.1), xycoords='axes fraction', weight='bold')

    ax.plot(
        estimated_mode,
        float(norm.pdf(estimated_mode, estimated_mode, estimated_fim)) * mode_scale_factor,
        color='#d55e00', marker='v', markersize=11,
        label='MLE')

    ax.plot(
        estimated_mean,
        float(norm.pdf(estimated_mean, estimated_mean, estimated_std)),
        color='#56b4e9', marker='o', markersize=11,
        label='MCMC')

    if ind == 0:
        ax.legend()

    ax.set_xlim(-5, 5)

fig.tight_layout()
fig.savefig('/tmp/figure_2.pdf', bbox_inches='tight', pad_inches=0)
# plt.show()