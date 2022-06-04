#!/usr/bin/env python

# Initial Date: May 2022
# Author: Rahul Bhadani
# Copyright (c)  Rahul Bhadani
# All rights reserved.

# Requires Python 3.8 or above

import math
from pyrsistent import v
from scipy.stats import rv_discrete
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import matplotlib as mpl
import pyplot_themes as themes
themes.theme_few(scheme="dark", grid=False, ticks=False,)
mpl.rcParams['font.family'] = 'Serif'


x1k = np.arange(1,21)
y1k = np.repeat(1/20, 20) #uniform discrete
pmf1 = rv_discrete(name='uniform discrete', values=(x1k, y1k))


# hypergeometric
N = 100
M = 50
K = 10
x2k = np.arange(min(M,K))
y2k = np.zeros(min(M,K))
for i, x in enumerate(x2k):
    y2k[i] = (math.comb(M, x)*math.comb(N-M, K-x))/math.comb(N, K)
# because of numerical round off and how many samples we choose, sum of y2k is less than 1. So just normalize it
y2k =y2k/sum(y2k)
pmf2 = rv_discrete(name='hypergeometric', values=(x2k, y2k))

# Binomial
p = 0.4
n = 20
x3k = np.arange(n)
y3k = np.zeros(n)
for i, x in enumerate(x3k):
    y3k[i] = math.comb(n, x)*(p**x)*((1-p)**(n-x))
y3k =y3k/sum(y3k)
pmf3 = rv_discrete(name='binomial', values=(x3k, y3k))

# Poisson
L = 4
n = 50
x4k = np.arange(n)
y4k = np.zeros(n)
for i, x in enumerate(x4k):
    y4k[i] =(np.exp(-L)*(L**x))/math.factorial(x)
y4k =y4k/sum(y4k)
pmf4 = rv_discrete(name='Poisson', values=(x4k, y4k))

# Geometric
p = 0.4
n = 20
x5k = np.arange(1,n+1)
y5k = np.zeros(n)
for i, x in enumerate(x5k):
    y5k[i] = p*((1-p)**(x-1))
y5k =y5k/sum(y5k)
pmf5 = rv_discrete(name='geometric', values=(x5k, y5k))

# Negative Binomial
p = 0.4
r = 10
n = 40
x6k = np.arange(n)
y6k = np.zeros(n)
for i, x in enumerate(x6k):
    y6k[i] = math.comb(r+x-1, r-1)*(p**r)*((1-p)**x)

y6k =y6k/sum(y6k)
pmf6 = rv_discrete(name='negative binomial', values=(x6k, y6k))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3)
ax=np.ravel(ax)

ax[0].plot(x1k, pmf1.pmf(x1k), 'o', ms=6, mec='#2F9C95', markerfacecolor="#2F9C95")
ax[0].vlines(x1k, 0, pmf1.pmf(x1k), colors='#2F9C95', lw=1)
ax[0].set_title('Uniform Discrete')

ax[1].plot(x2k, pmf2.pmf(x2k), 'o', ms=6, mec='#2F9C95', markerfacecolor="#2F9C95")
ax[1].vlines(x2k, 0, pmf2.pmf(x2k), colors='#2F9C95', lw=1)
ax[1].set_title('Hypergeometric')

ax[2].plot(x3k, pmf3.pmf(x3k), 'o', ms=6, mec='#2F9C95', markerfacecolor="#2F9C95")
ax[2].vlines(x3k, 0, pmf3.pmf(x3k), colors='#2F9C95', lw=1)
ax[2].set_title('Binomial')

ax[3].plot(x4k, pmf4.pmf(x4k), 'o', ms=6, mec='#2F9C95', markerfacecolor="#2F9C95")
ax[3].vlines(x4k, 0, pmf4.pmf(x4k), colors='#2F9C95', lw=1)
ax[3].set_title('Poisson')

ax[4].plot(x5k, pmf5.pmf(x5k), 'o', ms=6, mec='#2F9C95', markerfacecolor="#2F9C95")
ax[4].vlines(x5k, 0, pmf5.pmf(x5k), colors='#2F9C95', lw=1)
ax[4].set_title('Geometric')

ax[5].plot(x6k, pmf6.pmf(x6k), 'o', ms=6, mec='#2F9C95', markerfacecolor="#2F9C95")
ax[5].vlines(x6k, 0, pmf6.pmf(x6k), colors='#2F9C95', lw=1)
ax[5].set_title('Negative Binomial')

plt.tight_layout()
plt.show()