# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:53:29 2024

@author: Deep_
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the input data
x = np.loadtxt(r"C:\Users\Deep_\Downloads\Kaggle\Stochastic\input_data.txt")

# Normal distribution
m = np.mean(x)
v = np.var(x)
print('Normal distribution (MM/MLE)')
print(f'm = {m}, s2 = {v} (s = {np.sqrt(v)})')
print(f"1%: {stats.norm.ppf(0.01, m, np.sqrt(v))}, 99%: {stats.norm.ppf(0.99, m, np.sqrt(v))} m^3/s")

# Probability Plot with 90% KS Bounds
i = np.arange(1, len(x) + 1)
n = len(x)
q = stats.norm.ppf((i - 3/8) / (n + 1/4), m, np.sqrt(v))

# 90% KS Bounds (LB Table 7.5)
ca = 0.819 / (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n))
ub = stats.norm.ppf((i - 1) / n + ca, m, np.sqrt(v))
lb = stats.norm.ppf(i / n - ca, m, np.sqrt(v))

# Plot the probability plot with bounds
plt.figure(figsize=(8, 6))
stats.probplot(x, dist="norm", plot=plt)
plt.plot(sorted(q), sorted(x), 'o', label='Data')
plt.plot(sorted(q), lb, 'r--', label='Lower Bound (90%)')
plt.plot(sorted(q), ub, 'g--', label='Upper Bound (90%)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Probability Plot with 90% KS Bounds (Normal)')
plt.legend()
plt.show()

# Print PPCC (Probability Plot Correlation Coefficient)
r = np.corrcoef(np.sort(x), q)
print(f'PPCC: {r[0, 1]}')



