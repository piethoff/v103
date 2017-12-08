import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants.constants as const
from uncertainties import ufloat
from uncertainties import unumpy


mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

data = np.genfromtxt('content/rund_einseit.txt', unpack=True)

x = 0.5 * (data[0] /100)**2 - (data[0] / 300)**3
y= (data[2] -data[1]) /1000

plt.xlabel(r'$g(x)/\si{\cubic\meter}$')
plt.ylabel(r'$D(x)/\si{\meter}$')
plt.grid(True, which='both')


# Fitvorschrift
def f(x, A, B):
    return A*x + B      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(f, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

plt.plot(x, f(x, *params), "xkcd:orange", label=r'Regression' )
plt.plot(x, y, ".", color="xkcd:blue", label="Messwerte")



plt.tight_layout()
plt.legend()
plt.savefig('build/r_einseit.pdf')
plt.clf()

data = np.genfromtxt('content/eckig_einseit.txt', unpack=True)

x = 0.5 * (data[0] /100)**2 - (data[0] / 300)**3
y= (data[2] -data[1]) /1000

plt.xlabel(r'$g(x)/\si{\cubic\meter}$')
plt.ylabel(r'$D(x)/\si{\meter}$')
plt.grid(True, which='both')


# Fitvorschrift
def f(x, A, B):
    return A*x + B      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(f, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

plt.plot(x, f(x, *params), "xkcd:orange", label=r'Regression' )
plt.plot(x, y, ".", color="xkcd:blue", label="Messwerte")



plt.tight_layout()
plt.legend()
plt.savefig('build/e_einseit.pdf')
plt.clf()
