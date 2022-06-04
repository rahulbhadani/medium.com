import math
from pyrsistent import v
from scipy.stats import rv_continuous
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import matplotlib as mpl
import pyplot_themes as themes
from scipy.special import gamma, factorial
themes.theme_few(scheme="dark", grid=False, ticks=False)
mpl.rcParams['font.family'] = 'Serif'
s.set_context('paper')


class Uniform(rv_continuous):
    "Uniform Continuous"
    def _pdf(self, x):
        y = 0
        if (x < (self.a)):
            y = 0
        elif (x > (self.b)):
            y = 0
        else:
            y= 1.0/(self.b-self.a)
        return y

class Exponential(rv_continuous):
    "Exponential"
    def __init__(self, beta, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta


    def _pdf(self, x):
        if(x <=0):
            return 0

        y = (1.0/self.beta)*np.exp(-x/self.beta)
        return y

class Gamma(rv_continuous):
    "Gamma"
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def _pdf(self, x):
        if(x <=0):
            return 0

        y = (1.0/((self.beta**self.alpha)*gamma(self.alpha)))* np.exp(-x/self.beta)*(x**(self.alpha-1))
        return y

class Weibull(rv_continuous):
    "Weibull"
    def __init__(self, gamma, beta, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.beta = beta

    def _pdf(self, x):
        if(x <=0):
            return 0

        y = (self.gamma/self.beta)*(x**(self.gamma-1))*np.exp(-(x**self.gamma)/self.beta)
        return y

class Gaussian(rv_continuous):
    "Gaussian"
    def __init__(self, mu, sigma, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

    def _pdf(self, x):

        y = (1.0/(np.sqrt(2*np.pi)*self.sigma))*np.exp( - ( (x-self.mu)**2  )/(2*self.sigma*self.sigma)   )
        return y

# def BetaFunc(alpha, beta):
#     y = (gamma(alpha)*gamma(beta))/(gamma(alpha+beta))
#     return y

# class Beta(rv_continuous):
#     "Beta"
#     def __init__(self, alpha, beta, **kwargs):
#         super().__init__(**kwargs)
#         self.alpha = alpha
#         self.beta = beta

#     def _pdf(self, x):

#         if( x >=1):
#             return 0
#         if(x <=0):
#             return 0

#         num1 = x**(self.alpha-1)
#         num2 = (1-x)**(self.beta-1)
#         denom = BetaFunc(self.alpha, self.beta)
#         num = num1*num2
#         y = num/denom
#         return y

# class Cauchy(rv_continuous):
#     "Cauchy"
#     def __init__(self, theta, **kwargs):
#         super().__init__(**kwargs)
#         self.theta = theta

#     def _pdf(self, x):
#         y = (1.0/np.pi)*(1.0/(1 + ((x-self.theta)**2)))
#         return y

class Laplace(rv_continuous):
    "Laplace"
    def __init__(self, mu, sigma, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

    def _pdf(self, x):
        y = (1.0/(2*self.sigma))*np.exp(-abs(x-self.mu)/self.sigma)
        return y



P1 = Uniform(name='Uniform', a= 0, b =1)
B1 = P1.rvs(size = 1000)

P2 = Exponential(name='Exponential', beta = 2.0)
B2 = P2.rvs(size = 1000)

P3 = Gamma(name='Gamma', alpha = 1, beta = 0.5)
B3 = P3.rvs(size = 1000)

P4 = Weibull(name='Weibull', gamma = 0.5, beta =1.0)
B4 = P4.rvs(size = 1000)

P5 = Gaussian(name='Gaussian', mu = 0, sigma =1.0)
B5 = P5.rvs(size = 1000)

P6 = Laplace(name='Laplace', mu = 0, sigma =1.0)
B6 = P6.rvs(size = 1000)


fig, ax = plt.subplots(2, 3)
ax=np.ravel(ax)
s.histplot(B1, ax = ax[0], stat = 'density')
ax[0].set_xlabel('x',  fontsize = 20)
ax[0].set_xlim([0, 1])
ax[0].set_ylabel('f(x)', fontsize =20)
ax[0].set_title('Uniform Continuous', fontsize =20)
s.histplot(B2, ax = ax[1], stat = 'density')
ax[1].set_xlim([0, 15])
ax[1].set_xlabel('x',  fontsize = 20)
ax[1].set_ylabel('f(x)', fontsize =20)
ax[1].set_title('Exponential', fontsize =20)

s.histplot(B3, ax = ax[2], stat = 'density')
ax[2].set_xlim([0, 15])
ax[2].set_xlabel('x',  fontsize = 20)
ax[2].set_ylabel('f(x)', fontsize =20)
ax[2].set_title('Gamma', fontsize =20)

s.histplot(B4, ax = ax[3], stat = 'density')
ax[3].set_xlim([0, 15])
ax[3].set_xlabel('x',  fontsize = 20)
ax[3].set_ylabel('f(x)', fontsize =20)
ax[3].set_title('Weibull', fontsize =20)

s.histplot(B5, ax = ax[4], stat = 'density')
ax[4].set_xlim([-15, 15])
ax[4].set_xlabel('x',  fontsize = 20)
ax[4].set_ylabel('f(x)', fontsize =20)
ax[4].set_title('Gaussian', fontsize =20)

# s.histplot(B6, ax = ax[1], stat = 'density')
# ax[5].set_xlim([0, 1])
# ax[5].set_xlabel('x',  fontsize = 20)
# ax[5].set_ylabel('f(x)', fontsize =20)
# ax[5].set_title('Beta', fontsize =20)

# s.histplot(B7, ax = ax[1], stat = 'density')
# ax[6].set_xlim([-15, 15])
# ax[6].set_xlabel('x',  fontsize = 20)
# ax[6].set_ylabel('f(x)', fontsize =20)
# ax[6].set_title('Cauchy', fontsize =20)

s.histplot(B6, ax = ax[5], stat = 'density')
ax[5].set_xlim([-15, 15])
ax[5].set_xlabel('x',  fontsize = 20)
ax[5].set_ylabel('f(x)', fontsize =20)
ax[5].set_title('Laplace', fontsize =20)

plt.tight_layout()
# plt.xlabel('$x$', color = '#1C2541')
# plt.ylabel('f(x)',  color = '#1C2541')
# plt.show()

# plt.show()