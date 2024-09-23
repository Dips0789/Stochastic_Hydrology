# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:16:19 2024

@author: Deep_
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from scipy.optimize import fsolve
from scipy import stats


# Load data
x = np.loadtxt(r"C:\Users\Deep_\Downloads\Kaggle\Stochastic\input_data.txt")
i = np.arange(1, len(x) + 1)
n = len(x)

# Gumbel MM (Method of Moments)
print('Gumbel MM')
ahat = np.sqrt(np.var(x)) * np.sqrt(6) / np.pi
zhat = np.mean(x) - 0.5772 * ahat
q1 = zhat - ahat * np.log(-np.log(0.01))
q99 = zhat - ahat * np.log(-np.log(0.99))
print(f'alpha = {ahat}, xi = {zhat}')
print(f'1%: {q1}, 99%: {q99} m^3/s')
q = zhat - ahat * np.log(-np.log((i - 3/8) / (n + 1/4)))
r = np.corrcoef(np.sort(x), q)[0, 1]
print(f'PPCC: {r}')
print()

# Gumbel LM (L-Moments)
print('Gumbel L-Moments')
xs = np.sort(x)
suml = sum((j - 1) * xs[j] for j in range(1, n))
l2hat = 2 / (n * (n - 1)) * suml - np.mean(x)
ahat_lm = l2hat / np.log(2)
zhat_lm = np.mean(x) - 0.5772 * ahat_lm
q1_lm = zhat_lm - ahat_lm * np.log(-np.log(0.01))
q99_lm = zhat_lm - ahat_lm * np.log(-np.log(0.99))
print(f'alpha = {ahat_lm}, xi = {zhat_lm}')
print(f'1%: {q1_lm}, 99%: {q99_lm} m^3/s')
q_lm = zhat_lm - ahat_lm * np.log(-np.log((i - 3/8) / (n + 1/4)))
r_lm = np.corrcoef(np.sort(x), q_lm)[0, 1]
print(f'PPCC: {r_lm}')
print()

# Gumbel MLE (Maximum Likelihood Estimation)
def gumbel_mle(params):
    ahat_mle, zhat_mle = params
    term1 = (x - zhat_mle) / ahat_mle
    term2 = np.exp(-(x - zhat_mle) / ahat_mle)
    dL_dalpha = -n/ahat_mle + np.sum((term1 + 1) * term2)
    dL_dxi = -n/ahat_mle + np.sum(term2)
    return [dL_dxi, dL_dalpha]

print('Gumbel MLE')
ahat_mle, zhat_mle = fsolve(gumbel_mle, [600, 600])
q1_mle = zhat_mle - ahat_mle * np.log(-np.log(0.01))
q99_mle = zhat_mle - ahat_mle * np.log(-np.log(0.99))
print(f'alpha = {ahat_mle}, xi = {zhat_mle}')
print(f'1%: {q1_mle}, 99%: {q99_mle} m^3/s')
q_mle = zhat_mle - ahat_mle * np.log(-np.log((i - 3/8) / (n + 1/4)))
r_mle = np.corrcoef(np.sort(x), q_mle)[0, 1]
print(f'PPCC: {r_mle}')
print()

# 90% KS Bounds (LB Table 7.5)
ca = 0.819 / (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))
ub = zhat_mle - ahat_mle * np.log(-np.log((i - 1) / n + ca))
lb = zhat_mle - ahat_mle * np.log(-np.log(i / n - ca))
lb[np.imag(lb) != 0] = np.nan  # Avoid negative logs resulting in complex numbers
ub[np.imag(ub) != 0] = np.nan

# Probability plot
plt.figure(figsize=(8, 6))
stats.probplot(x, dist="gumbel_r", sparams=(zhat_mle, ahat_mle), plot=plt)
plt.plot(np.sort(q_mle), np.sort(x), 'o', label='Data')
plt.plot(np.sort(q_mle), lb, 'r--', label='Lower Bound (90%)')
plt.plot(np.sort(q_mle), ub, 'g--', label='Upper Bound (90%)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Probability Plot with 90% KS Bounds (Gumbel)')
plt.legend()
plt.show()
