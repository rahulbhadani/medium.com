#!/usr/bin/env python

# Initial Date: May 2022
# Author: Rahul Bhadani
# Copyright (c)  Rahul Bhadani
# All rights reserved.

from scipy.stats import rv_continuous
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import matplotlib as mpl
import pyplot_themes as themes
themes.theme_paul_tol( grid=False, ticks=False,)
mpl.rcParams['font.family'] = 'Serif'

class Uniform(rv_continuous):
    "Uniform distribution"
    def _pdf(self, x):
        y = 0
        if (x < (self.a)):
            y = 0
        elif (x > (self.b)):
            y = 0
        else:
            y= 1.0/(self.b-self.a)
        return y
P = Uniform(name='Uniform', a= 0, b =1)
B = P.rvs(size = 10000)
s.distplot(B)
plt.xlabel('$x$', color = '#1C2541')
plt.ylabel('f(x)',  color = '#1C2541')
plt.show()

print('Second moment of the uniform distribution is {}'.format(P.moment(n=2)))