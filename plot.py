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
y= (data[1] -data[2]) /1000

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
y= (data[1] -data[2]) /1000

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



data = np.genfromtxt("content/eckig_beidseit.txt", unpack=True)
data[0] /= 100
data[1] /= 1000
data[2] /= 1000
L = 0.55
mitte = data[0][int(data[0].size/2)]

def g(x, A):
    y = np.array([])
    for i in x:
        if(i <= mitte):
            y = np.append(y, A*(3*L**2*i-4*i**3))
        else:
            y = np.append(y, A*(4*i**3-12*L*i**2+9*L**2*i-L**3))
    return y

data[1] -= data[2]
x= np.linspace(data[0][0]-0.05, data[0][-1]+0.05, 1000)

plt.plot(data[0], data[1], ".", color="xkcd:blue", label="Messwerte")

params, covar = curve_fit(g, data[0], data[1])
plt.plot(x, g(x, *params), color="xkcd:orange", label=r"Regression")
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
print("Beidseitige Einspannung:", uparams)

plt.ylabel(r"$D(x)/\si{\meter}$")
plt.xlabel(r"$x/\si{\meter}$")

plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig("build/e_beidseit.pdf")
plt.clf()