# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:11:37 2024

@author: Deep_
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load data
x = np.loadtxt(r"C:\Users\Deep_\Downloads\Kaggle\Stochastic\input_data.txt")
i = np.arange(1, len(x) + 1)
n = len(x)

# 3-parameter lognormal: Quantile lower bound estimator (LB 7.81)
that = (min(x) * max(x) - np.median(x) ** 2) / (min(x) + max(x) - 2 * np.median(x))

print('3-parameter lognormal distribution, Log-space estimates')
mhat = np.mean(np.log(x - that))
vhat = np.mean((np.log(x - that) - mhat) ** 2)
print(f'm = {mhat}, s2 = {vhat}, t = {that} (s = {np.sqrt(vhat)})')
print(f"1%: {stats.lognorm.ppf(0.01, np.sqrt(vhat), scale=np.exp(mhat)) + that} m^3/s")
print(f"99%: {stats.lognorm.ppf(0.99, np.sqrt(vhat), scale=np.exp(mhat)) + that} m^3/s")
q = stats.lognorm.ppf((i - 3/8) / (n + 1/4), np.sqrt(vhat), scale=np.exp(mhat)) + that
r = np.corrcoef(np.sort(x), q)[0, 1]
print(f'PPCC: {r}')
print()

# 3-parameter lognormal: Real-space estimates (LB 7.82)
print('3-parameter lognormal distribution, Real-space estimates')
mhat_real = np.log((np.mean(x) - that) / np.sqrt(1 + np.var(x) / ((np.mean(x) - that) ** 2)))
vhat_real = np.log(1 + np.var(x) / ((np.mean(x) - that) ** 2))
print(f'm = {mhat_real}, s2 = {vhat_real}, t = {that} (s = {np.sqrt(vhat_real)})')
print(f"1%: {stats.lognorm.ppf(0.01, np.sqrt(vhat_real), scale=np.exp(mhat_real)) + that} m^3/s")
print(f"99%: {stats.lognorm.ppf(0.99, np.sqrt(vhat_real), scale=np.exp(mhat_real)) + that} m^3/s")
q_real = stats.lognorm.ppf((i - 3/8) / (n + 1/4), np.sqrt(vhat_real), scale=np.exp(mhat_real)) + that
r_real = np.corrcoef(np.sort(x), q_real)[0, 1]
print(f'PPCC: {r_real}')
print()

# 90% KS Bounds (LB Table 7.5)
ca = 0.819 / (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))
ub = stats.lognorm.ppf((i - 1) / n + ca, np.sqrt(vhat_real), scale=np.exp(mhat_real)) + that
lb = stats.lognorm.ppf(i / n - ca, np.sqrt(vhat_real), scale=np.exp(mhat_real)) + that

# Plot probability plot with bounds
plt.figure(figsize=(8, 6))
stats.probplot(x, dist="lognorm", sparams=(np.sqrt(vhat_real), 0, np.exp(mhat_real)), plot=plt)
plt.plot(np.sort(q_real), np.sort(x), 'o', label='Data')
plt.plot(np.sort(q_real), lb, 'r--', label='Lower Bound (90%)')
plt.plot(np.sort(q_real), ub, 'g--', label='Upper Bound (90%)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Probability Plot with 90% KS Bounds (Lognormal-3)')
plt.legend()
plt.show()
