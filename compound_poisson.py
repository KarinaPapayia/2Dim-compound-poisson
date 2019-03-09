import numpy as np
from math import exp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
import matplotlib.cm as cm
from pylab import figure, plot, show, grid, axis, xlabel, ylabel, title, hold, draw
import scipy.stats as stats
import seaborn as sns
from sympy import plot_implicit
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import mean_squared_error


#generation of the poisson proces with lambda=10, 12 ( total number of jumps on [0,1]).
N1 = np.random.poisson(10)

#generation of N r.v from uniform distribution for the jump times
u1 = np.random.uniform(0, 1, N1)
u1.sort()

#time
ts = np.linspace (0, 1, 1000)
#n = 1000

#generation of two dimensional normal random variable for the jump size using box-muller transformation
def box_muller(n):
    """Generate n random standard normal bivariate with box-muller transformation."""

    u1 = np.random.random((n+1)//2) #floor division returns integer instead of float number
    u2 = np.random.random((n+1)//2)
    r_squared = -2*np.log(u1)
    r = np.sqrt(r_squared)
    theta = 2*np.pi*u2
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.empty(n)
    z[:((n+1)//2)] = x
    z[((n+1)//2):] = y
    return z[:n]

#mean of the jump size
mu_jumps = np.zeros(2)

#covariance matrix for the jump size
C_jumps = np.array([[4, 1], [1, 6]])

def mvn(mu, sigma, n=1):
    """Generate n samples from bivariate normal with mean mu and covariance sigma."""

    A = np.linalg.cholesky(sigma) #cholesky decomposition for covariance matrix
    p = len(mu)

    zs = np.zeros((n, p))
    for i in range(n):
        z = box_muller(p)
        zs[i] = mu + A@z
    return zs

#x, y = mvn(mu, sigma, n).T
jumps = mvn(mu_jumps, C_jumps, N1).T
J = np.array(jumps)
#g = sns.jointplot(x, y, kind='scatter')
pass
pass

#creat the vector 1(U_i < t) for i =1,...N (poisson)
def func(u1_elmnt, ts_elmnt):
    if u1_elmnt < ts_elmnt:
        return 1
    return 0

l = [[func(x, y) for x in u1] for y in ts]
time = np.array(l)

#simulation of two dimensional compound poisson process with same number of jumps
def IncrementsCompoundPoissonTwoDim(ts):
    compound_2dim = np.zeros((len(ts), 2))
    jumps = mvn(mu_jumps, C_jumps, N1).T
    l = [[func(x, y) for x in u1] for y in ts]
    time = np.array(l)
    product = [np.multiply(time, jumps[i]) for i in range(2)]
    compound_2dim = [product[i].sum(axis = 1) for i in range(2)]
    return compound_2dim
Compound2Dim = IncrementsCompoundPoissonTwoDim(ts)
print(np.shape(Compound2Dim))
#for i in range(2):
#    plt.plot(ts, compound_2dim[i])
#plt.plot(ts, compound2[:,0], drawstyle ='steps-pre')
#plt.plot(ts, compound2[:,1], drawstyle = 'steps-pre')
#plt.tight_layout()

#plt.show()
