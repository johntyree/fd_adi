#!/usr/bin/env python
"""Description."""

# import sys
# import os
# import itertools as it

import scipy.integrate
from pylab import *


I = 1j

# Missing:
    # We can't set mean_reversion, kappa: assumed 1?
    # We can't set long term mean, theta: v0
class HestonCos(object):
    def __init__(self, S, K, r, vol, T, kappa, theta, sigma, rho, N=2**8):
        r,vol,T,kappa,theta,sigma,rho = map(float, [r,vol,T,kappa, theta, sigma,rho])
        if hasattr(S, '__iter__') and hasattr(K, '__iter__'):
            raise TypeError("Can only have array(K) or array(S), not both.")
        elif hasattr(K, '__iter__'):
            K = array([K], copy=False, dtype=float)
            S = array([S], dtype=float)
        elif hasattr(S, '__iter__'):
            S = array(S, copy=False, dtype=float)
            K = array([K], dtype=float)
        else:
            K = array([K], dtype=float)
        self.S     = S
        self.K     = K
        self.r     = r
        self.vol   = vol
        self.T     = T
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho   = rho
        self.N = N

    def solve(self):
        return self.COS(self.S,
                        self.K,
                        self.r,
                        self.vol**2,
                        self.T,
                        self.kappa,
                        self.theta,
                        self.rho,
                        self.sigma,
                       )


    def xi(self, k,a,b,c,d):
        return (
            1./(1+(k*pi/(b-a))**2) *
            (cos(k*pi*(d-a)/(b-a))
             * exp(d)
             - cos(k*pi*(c-a)/(b-a))
             * exp(c)
             + k*pi/(b-a)
             * sin(k*pi*(d-a)/(b-a))
             * exp(d)
             - k*pi/(b-a)
             * sin(k*pi*(c-a)/(b-a))
             * exp(c))
        )

    def psi(self, k, a, b, c, d):
        if not hasattr(k, '__iter__'):
            print type(k), k
            k = array([k])
        ret = hstack([d-c,
                     (sin(k[1:]*pi*(d-a) / (b-a))
                      - sin(k[1:]*pi*(c-a)
                            / (b-a)
                           )
                     )
                     * (b-a)
                     / (k[1:] * pi)
                    ])
        return ret

    def CF(self, omega, r, var, T, kappa, theta, sigma, rho):
        D = sqrt((kappa - 1j*rho*sigma*omega)**2 + (omega**2 + 1j*omega)*sigma**2)
        G = (kappa - 1j*rho*sigma*omega - D) / (kappa - 1j*rho*sigma*omega + D)
        return (exp(1j * omega * r * T + var/sigma**2 *
            ((1-exp(-D*T))/(1-G*exp(-D*T)))*(kappa - 1j*rho*sigma*omega - D))
            * exp((kappa*theta)/sigma**2 * (T*(kappa - 1j*rho*sigma*omega - D) -
                                          2*log((1-G*exp(-D*T))/(1-G)))))

    def COS(self, S, K, r, var, T, kappa, theta, rho, sigma):
        N = self.N
        L = 10
        c1 = r*T + (1 - exp(-kappa*T))*(theta-var/(2*kappa)) - 0.5*theta*T
        c2 = 0.125*kappa**(-3) * (sigma*T*kappa*exp(-kappa*T)*(var-theta)*(8*kappa*rho - 4*sigma)
            + kappa*rho*sigma*(1-exp(-kappa*T))*(16*theta - 8*var)
            + 2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)
            + sigma**2*((theta - 2*var) * exp(-2*kappa*T) + theta*(6*exp(-kappa*T) - 7) + 2*var)
            + 8*kappa**2*(var-theta)*(1-exp(-kappa*T)))
        c4 = 0

        a = c1-L*sqrt(abs(c2)+sqrt(c4))
        b = c1+L*sqrt(abs(c2)+sqrt(c4))
        x = log(S/K)
        k = arange(N)

        U = 2./(b-a)*(self.xi(k,a,b,0,b) -self.psi(k,a,b,0,b))

        unit = hstack((.5, ones(N-1)))

        CFVG_tiled = tile(
            self.CF(k*pi/(b-a), r, var, T, kappa, theta, sigma, rho)[:,newaxis],
            (1,len(K)))
        ret = unit.dot(
            (CFVG_tiled * exp(1j*k[:,newaxis]*pi*(x-a)/(b-a)))
            * tile(U[:,newaxis], (1, len(K))))
        ret= K * exp(-r*T) * ret.real;
        return ret




hc = HestonCos(100,99,0.06, 0.2, 1, 1, 0.2, 0.001, 0)


class HestonFundamental(object):

    def __init__(self, s, k, r, v, t, kappa, theta, sigma, rho):
        # if hasattr(s, '__iter__') and hasattr(k, '__iter__'):
            # raise TypeError("Can only have array(k) or array(s), not both.")
        # elif hasattr(k, '__iter__'):
            # k = array(k, copy=False)
        # elif hasattr(s, '__iter__'):
            # s = array(s, copy=False)
        # else:
            # k = array(k,)
        self.spot = s
        self.logspot = log(s)
        self.strike = k
        self.r = r
        self.var = v**2
        self.rho = rho
        self.sig = sigma
        self.tenor = t
        self.mean_reversion = kappa
        self.mean_variance = theta
        self.lam = 0
        self.b = None
        self.u = None
        self.dmemo = None
        self.gmemo = None


    def d(self, phi):
        if not self.dmemo:
            left  = pow((self.rho * self.sig * phi * I - self.b),2)
            right = pow(self.sig,2) * (2 * self.u * phi * I - pow(phi,2))
            self.dmemo = sqrt(left-right)
        return self.dmemo

    def g(self, phi):
        if not self.gmemo:
            self.gmemo = (self.b - (self.rho * self.sig * phi * I) + self.d(phi)) \
                 / (self.b - (self.rho * self.sig * phi * I) - self.d(phi))
        return self.gmemo


    def C(self, phi):
        return (self.r * phi * I * self.tenor
             + (self.mean_reversion * self.mean_variance / pow(self.sig,2))
             * ((self.b - self.rho * self.sig * phi * I + self.d(phi)) * self.tenor
             - 2.0 * log((1.0 - self.g(phi) * exp(self.d(phi) * self.tenor))
                     / (1.0 - self.g(phi)))))


    def D(self, phi):
        return ((self.b - self.rho * self.sig * phi * I + self.d(phi)) / pow(self.sig,2)
               * ((1.0 - exp(self.d(phi) * self.tenor))
                / (1.0 - self.g(phi) * exp(self.d(phi) * self.tenor))))


    def f(self, phi):
        return exp(self.C(phi) + self.D(phi)*self.var + I * phi * self.logspot)


    def integrand(self, phi):
        self.dmemo = self.gmemo = None
        return (exp(-I * phi * log(self.strike)) * self.f(phi) / (I * phi)).real

    def ugly_integrate(self, phi, minR, maxR, neval):
        stepR = float(maxR-minR) / neval
        sum = 0
        for i in xrange(neval):
            x = minR + stepR * i;
            tmp = self.integrand(x)
            sum += tmp
        ret = sum
        return ret * stepR

    def P(self):
        # res.append(scipy.integrate.quad(self.integrand, 5e-2, inf)[0] * 1./pi + 0.5)
        res = self.ugly_integrate(self.integrand, 5e-2, 100, 1000)
        return res * 1./pi + 0.5

    def solve(self):
        self.u = 0.5
        self.b = self.mean_reversion + self.lam - self.rho * self.sig
        P1 = self.P()

        self.u = -0.5
        self.b = self.mean_reversion + self.lam
        P2 = self.P()

        discount = exp(-self.r * self.tenor)

        return self.spot * P1 - self.strike * P2 * discount


def call(s=100, k=99, r=0.06, vol=0.2, t=1, kappa=1, theta=None,
         sigma=1e-4, rho=0):
    if theta is None:
        theta = vol**2
    # return HestonSolver(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
    a,b = 0, 0
    try:
        a =  HestonFundamental(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
    except Exception, e:
        print "Fundamental failed:", e
    b = HestonCos(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
    try:
        b =  HestonCos(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
    except TypeError, e:
        print "COS failed:", e
    return (a,b, b-a)

# print call(s=array((90,100,110)), k=90)
print call(s=array((90,100,110)), k=99)
# print call(s=array((90,100,110)), k=110)
# print call(s=110, k=array((90,100,110)))
# print call(s=100, k=array((90,99,110)))
# print call(s=90, k=array((90,100,110)))

#PLot for v= 0, v = 0.1, etc
