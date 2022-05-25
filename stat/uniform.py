#!/usr/bin/env python

# Initial Date: May 2022
# Author: Rahul Bhadani
# Copyright (c)  Rahul Bhadani
# All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import matplotlib as mpl
import pyplot_themes as themes
themes.theme_solarized(scheme="dark", grid=False, ticks=False,)
mpl.rcParams['font.family'] = 'Serif'

A = np.random.uniform(low = 0, high = 2, size = 10000)
s.set_context('talk')
s.distplot(A)
plt.xlabel('x', color = '#DAD6D6')
plt.ylabel('f(x)',  color = '#DAD6D6')
plt.title('U[0, 2]\n')
plt.show()

s.distplot(A**3)
plt.xlabel('$x^3$', color = '#DAD6D6')
plt.ylabel('f(x)',  color = '#DAD6D6')
plt.show()


from scipy.stats import rv_continuous
class YCube(rv_continuous):
    "YCube Power distribution"
    def _pdf(self, x):
        y = 0
        if (x < (self.a)):
            y = 0
        elif (x > (self.b)):
            y = 0
        else:
            y= (1.0/6.0)*(x**(-2.0/3.0))
        return y
P = YCube(name='pareto', a= 0, b =8)
B = P.rvs(size = 10000)
s.distplot(B)
plt.xlabel('$y$', color = '#DAD6D6')
plt.ylabel('f(y)',  color = '#DAD6D6')
plt.show()

