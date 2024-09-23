# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:08:33 2024

@author: Deep_
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load data
x = np.loadtxt(r"C:\Users\Deep_\Downloads\Kaggle\Stochastic\input_data.txt")
i = np.arange(1, len(x) + 1)
n = len(x)

# 2-parameter lognormal: MM (Method of Moments)
print('2-parameter lognormal distribution, MM')
vhat = np.log(1 + np.var(x) / (np.mean(x) ** 2))
mhat = np.log(np.mean(x)) - 0.5 * vhat
print(f'm = {mhat}, s2 = {vhat} (s = {np.sqrt(vhat)})')
print(f"1%: {stats.lognorm.ppf(0.01, np.sqrt(vhat), scale=np.exp(mhat))} m^3/s")
print(f"99%: {stats.lognorm.ppf(0.99, np.sqrt(vhat), scale=np.exp(mhat))} m^3/s")
q = stats.lognorm.ppf((i - 3/8) / (n + 1/4), np.sqrt(vhat), scale=np.exp(mhat))
r = np.corrcoef(np.sort(x), q)[0, 1]
print(f'PPCC: {r}')
print()

# 2-parameter lognormal: MLE (Maximum Likelihood Estimation)
print('2-parameter lognormal distribution, MLE')
mhat_mle = np.mean(np.log(x))
vhat_mle = np.mean((np.log(x) - mhat_mle) ** 2)
print(f'm = {mhat_mle}, s2 = {vhat_mle} (s = {np.sqrt(vhat_mle)})')
print(f"1%: {stats.lognorm.ppf(0.01, np.sqrt(vhat_mle), scale=np.exp(mhat_mle))} m^3/s")
print(f"99%: {stats.lognorm.ppf(0.99, np.sqrt(vhat_mle), scale=np.exp(mhat_mle))} m^3/s")
q_mle = stats.lognorm.ppf((i - 3/8) / (n + 1/4), np.sqrt(vhat_mle), scale=np.exp(mhat_mle))
r_mle = np.corrcoef(np.sort(x), q_mle)[0, 1]
print(f'PPCC: {r_mle}')
print()

# Probability Plot for MLE
# 90% KS Bounds (LB Table 7.5)
ca = 0.819 / (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))
ub = stats.lognorm.ppf((i - 1) / n + ca, np.sqrt(vhat_mle), scale=np.exp(mhat_mle))
lb = stats.lognorm.ppf(i / n - ca, np.sqrt(vhat_mle), scale=np.exp(mhat_mle))

# Plot probability plot with bounds
plt.figure(figsize=(8, 6))
stats.probplot(x, dist="lognorm", sparams=(np.sqrt(vhat_mle), 0, np.exp(mhat_mle)), plot=plt)
plt.plot(np.sort(q_mle), np.sort(x), 'o', label='Data')
plt.plot(np.sort(q_mle), lb, 'r--', label='Lower Bound (90%)')
plt.plot(np.sort(q_mle), ub, 'g--', label='Upper Bound (90%)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Probability Plot with 90% KS Bounds (Lognormal-2)')
plt.legend()
plt.show()

