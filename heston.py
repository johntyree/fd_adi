#!/usr/bin/env python
"""Description."""

# import sys
# import os
import itertools as it
import time

import scipy.stats
import scipy.integrate
from pylab import *


I = 1j

BLACKSCHOLES, FUNDAMENTAL, COS = [2**i for i in range(3)]
HESTON = FUNDAMENTAL | COS
ALL = BLACKSCHOLES | HESTON

class HestonCos(object):
    def __init__(self, S, K, r, vol, T, kappa, theta, sigma, rho, N=2**9):
        r,vol,T,kappa,theta,sigma,rho = map(float, [r,vol,T,kappa, theta, sigma,rho])
        S = array(S, dtype=float)
        K = array(K, dtype=float)
        if len(S.shape) > 0 and len(K.shape) > 0:
            if sum(S.shape) > len(S.shape) and sum(K.shape) > len(K.shape):
                raise TypeError("Can only have multiple spots or strikes, not both.")
        if len(S.shape) == 0:
            S = S[newaxis]
        # if len(S.shape) < 1:
            # S = S[:,newaxis]
        if len(K.shape) == 0:
            K = K[newaxis]
        # if len(K.shape) < 1:
            # K = K[:,newaxis]
        # if hasattr(K, '__iter__') and hasattr(K, '__iter__'):
            # raise TypeError("Can only have array(K) or array(S), not both.")
        # elif hasattr(K, '__iter__'):
            # K = array(K, copy=False, dtype=float)
            # S = array([S], dtype=float)
        # elif hasattr(S, '__iter__'):
            # S = array(S, copy=False, dtype=float)
            # K = array([K], dtype=float)
        # else:
            # K = array([K], dtype=float)
        # if not hasattr(S, '__iter__') and hasattr(K, '__iter__'):
            # raise TypeError("Can only have array(K) or array(S), not both.")

        self.S     = S
        self.K     = K
        self.r     = r
        self.vol   = array([vol])
        self.T     = T
        self.kappa = kappa
        self.theta = theta
        self.sigmama = sigma
        self.rho   = rho
        self.N = N

    def __str__(self):
        return '\n'.join([
        "Heston COS Object: %s" % id(self)
        ,"S    : %s" % self.S
        ,"K    : %s" % self.K
        ,"r    : %s" % self.r
        ,"vol  : %s" % self.vol
        ,"T    : %s" % self.T
        ,"kappa: %s" % self.kappa
        ,"theta: %s" % self.theta
        ,"sigma: %s" % self.sigmama
        ,"rho  : %s" % self.rho
        ,"     : %s" % self.N])



    def solve(self):
        return self.COS(self.S,
                        self.K,
                        self.r,
                        self.vol,
                        self.T,
                        self.kappa,
                        self.theta,
                        self.rho,
                        self.sigmama,
                       )


    def xi(self, k,a,b,c,d):
        return (
            1.0/(1+(k*pi/(b-a))**2) *
            (cos(k*pi*(d-a)/(b-a)) * exp(d)
             - cos(k*pi*(c-a)/(b-a)) * exp(c)
             + k*pi/(b-a) * sin(k*pi*(d-a)/(b-a)) * exp(d)
             - k*pi/(b-a) * sin(k*pi*(c-a)/(b-a)) * exp(c))
        )

    def psi(self, k, a, b, c, d):
        if not hasattr(k, '__iter__'):
            print type(k), k
            k = array([[k]])
        ret = vstack([d-c,
                      (sin(k[1:,:]*pi*(d-a) / (b-a))
                       - sin(k[1:,:]*pi*(c-a) / (b-a))
                      ) * (b-a) / (k[1:,:] * pi)
                    ])
        return ret

    def CF(self, omega, r, var, T, kappa, theta, sigma, rho):
        D = sqrt((kappa - 1j*rho*sigma*omega)**2 + (omega**2 + 1j*omega)*sigma**2);
        G = (kappa - 1j*rho*sigma*omega - D) / (kappa - 1j*rho*sigma*omega + D)
        p0 = 1j*omega*r*T
        p1 = var/sigma**2 * (1 - exp(-D*T))
        p2 = 1 - G * exp(-D*T)
        p3 = kappa - 1j*rho*sigma*omega-D
        p4 = kappa * theta / sigma**2
        phi = exp(p0 + p1 / p2 * p3) * exp(p4 * (T*p3 - 2*log(p2 / (1-G))))
        # print "p0:", p0, "\n"
        # print "p1:", p1, "\n"
        # print "p2:", p2, "\n"
        # print "p3:", p3, "\n"
        # print "p4:", p4, "\n"
        # print phi
        return phi

    def COS(self, S, K, r, vol, T, kappa, theta, rho, sigma):
        N = self.N
        L = 14

        # c1=r*T+(1-exp(-kappa*T))*(theta-vol)/(2*kappa)-0.5*theta*T;
        c1 = r*T + (1 - exp(-kappa*T))*(theta-vol)/(2*kappa) - 0.5*theta*T

        p1=sigma*T*kappa*exp(-kappa*T)*(vol-theta)*(8*kappa*rho-4*sigma);
        p2=kappa*rho*sigma*(1-exp(-kappa*T))*(16*theta-8*vol);
        p3=2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2);
        p4=sigma**2*((theta-2*vol)*exp(-2*kappa*T)+theta*(6*exp(-kappa*T)-7)+2*vol);
        p5=8*kappa**2*(vol-theta)*(1-exp(-kappa*T));
        c2=1/(8*kappa**3)*(p1+p2+p3+p4+p5);

        # c2 = 0.125*kappa**(-3) * (sigma*T*kappa*exp(-kappa*T)*(vol-theta)*(8*kappa*rho - 4*sigma)
            # + kappa*rho*sigma*(1-exp(-kappa*T))*(16*theta - 8*vol)
            # + 2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)
            # + sigma**2*((theta - 2*vol) * exp(-2*kappa*T) + theta*(6*exp(-kappa*T) - 7) + 2*vol)
            # + 8*kappa**2*(vol-theta)*(1-exp(-kappa*T)))

        #   Axis 0: Spots
        #   Axis 1: Cosines

        x = log(S/K)[newaxis, :]  # row
        a = x+c1-L*sqrt(abs(c2))  # row
        b = x+c1+L*sqrt(abs(c2))  # row
        k = arange(float(N))[:,newaxis]  # col
        omega=k*pi/(b-a)

        NPOINTS = max(len(S), len(K))


        U_tiled = 2./(b-a)*(self.xi(k,a,b,0,b) - self.psi(k,a,b,0,b))

        unit = hstack((.5, ones(N-1)))

        CF_tiled = self.CF(omega, r, vol**2, T, kappa, theta, sigma, rho)

        # CF_tiled = tile(
            # self.CF(omegar, vol**2, T, kappa, theta, sigma, rho)[:,newaxis],
            # (1,NPOINTS))
        # print U_tiled
        # print CF_tiled
        ret = unit.dot(
            (CF_tiled * exp(1j*k*pi*(x-a)/(b-a))) * U_tiled)
        ret = K * exp(-r*T) * ret.real;
        return ret




hc = HestonCos(array([90,100,110]),99,0.06, 0.2, 1, 1, 0.2, 0.001, 0)


class HestonFundamental(object):

    def __init__(self, S, K, r, vol, T, kappa, theta, sigma, rho):
        # if hasattr(s, '__iter__') and hasattr(k, '__iter__'):
            # raise TypeError("Can only have array(k) or array(s), not both.")
        # elif hasattr(k, '__iter__'):
            # k = array(k, copy=False)
        # elif hasattr(s, '__iter__'):
            # s = array(s, copy=False)
        # else:
            # k = array(k,)
        self.S = S
        self.logspot = log(S)
        self.K = K
        self.r = r
        self.var = vol**2
        self.rho = rho
        self.sigma = sigma
        self.T = T
        self.mean_reversion = kappa
        self.mean_variance = theta
        self.lam = 0
        self.b = None
        self.u = None
        self.dmemo = None
        self.gmemo = None

    def setvol(self, v):
        self.var = v**2
        self.vol = v

    property(None, setvol)

    def d(self, phi):
        if not self.dmemo:
            left  = pow((self.rho * self.sigma * phi * I - self.b),2)
            right = pow(self.sigma,2) * (2 * self.u * phi * I - pow(phi,2))
            self.dmemo = sqrt(left-right)
        return self.dmemo

    def g(self, phi):
        if not self.gmemo:
            self.gmemo = (self.b - (self.rho * self.sigma * phi * I) + self.d(phi)) \
                 / (self.b - (self.rho * self.sigma * phi * I) - self.d(phi))
        return self.gmemo


    def C(self, phi):
        return (self.r * phi * I * self.T
             + (self.mean_reversion * self.mean_variance / pow(self.sigma,2))
             * ((self.b - self.rho * self.sigma * phi * I + self.d(phi)) * self.T
             - 2.0 * log((1.0 - self.g(phi) * exp(self.d(phi) * self.T))
                     / (1.0 - self.g(phi)))))


    def D(self, phi):
        return ((self.b - self.rho * self.sigma * phi * I + self.d(phi)) / pow(self.sigma,2)
               * ((1.0 - exp(self.d(phi) * self.T))
                / (1.0 - self.g(phi) * exp(self.d(phi) * self.T))))


    def f(self, phi):
        return exp(self.C(phi) + self.D(phi)*self.var + I * phi * self.logspot)


    def integrand(self, phi):
        self.dmemo = self.gmemo = None
        return (exp(-I * phi * log(self.K)) * self.f(phi) / (I * phi)).real

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
        self.b = self.mean_reversion + self.lam - self.rho * self.sigma
        P1 = self.P()

        self.u = -0.5
        self.b = self.mean_reversion + self.lam
        P2 = self.P()

        discount = exp(-self.r * self.T)

        return self.S * P1 - self.K * P2 * discount


def hs_call(s, k, r, vol, t, kappa, theta, sigma, rho, HFUNC=HestonCos):
    if t == 0:
        t = 1e-6
    s = array(s);
    vol = array(vol)
    if len(vol.shape) == 0:
        vol = vol[newaxis]
    h = HFUNC(s, k, r, 0, t, kappa, 0, sigma, rho)
    if theta is not None:
        h.theta = theta
    ret = empty((len(h.S), len(vol)))
    for j in range(len(vol)):
        h.vol = vol[j]
        if theta is None:
            h.theta = h.vol**2
        ret[:,j] = h.solve()
    ret[isnan(ret)] = 0
    return ret


def hs_stream(s, k, r, v, dt, kappa, theta, sigma, rho, HFUNC=HestonCos):
    ret = empty((len(s), len(v)))
    h = HFUNC(s, k, r, 0, 0, kappa, theta, sigma, rho)
    yield bs_call_delta(s[:,newaxis], k, r, sqrt(var[newaxis,:]), 0)[0]
    for i in it.count(1):
        yield hs_call(s, k, r, v, i*dt, kappa, theta, sigma, rho)
        # h.T = i*dt
        # for j in range(len(v)):
            # h.vol = v[j]
            # ret[:,j] = h.solve()
        # # print h
        # yield ret

def bs_call_delta(s, k, r, vol, t):
    N = scipy.stats.distributions.norm.cdf
    s = array(np.maximum(1e-10, s), dtype=float)
    v = array(np.maximum(1e-10, vol), dtype=float)
    if len(s.shape) == 0: s = array([s])
    if len(v.shape) == 0: v = array([v])
    while len(s.shape) < 2:
        s = s[:,newaxis]
    while len(v.shape) < 2:
        v = v[newaxis,:]
    if t == 0:
        p = np.tile(s-k, len(v))
        d = array(p > 0, int)
        return (p, d)
    d1 = (np.log(s/k) + (r+0.5*v**2) * t) / (v * np.sqrt(t))
    d2 = d1 - v*np.sqrt(t)
    ret = (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))
    return ret

def call(s=100, k=99, r=0.06, vol=0.2, t=1, kappa=1, theta=None,
         sigma=1e-4, rho=0, method=COS, NCOS=2**8):
    if theta is None:
        theta = vol**2
    # s = array(s, dtype=float, copy=True)
    ret = []
    # return HestonSolver(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
    if method & BLACKSCHOLES:
        try:
            x = array(bs_call_delta(s, k, r, vol, t))[:1]
        except ValueError, e:
            print "BlackScholes failed:", e
            x = None
        ret.append(x)
    if method & FUNDAMENTAL:
        try:
            x = HestonFundamental(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
        except ValueError, e:
            print "Fundamental failed:", e
            x = None
        except TypeError, e:
            print "Fundamental failed:", e
            x = None
        ret.append(x)
    if method & COS:
        try:
            x = HestonCos(s,k,r,vol,t,kappa,theta,sigma,rho,N=NCOS).solve()
        except ValueError, e:
            print "COS failed:", e
            x = None
            raise e
        # except TypeError, e:
            # print "COS failed:", e
            # x = None
            # raise e
        ret.append(x)
    return ret

# print call(s=array((90,100,110)), k=90)
# print call(s=array((90,100,110)), k=99, method=ALL)
# print call(s=array((90,100,110)), k=110)
# print call(s=110, k=array((90,100,110)))
# print call(s=100, k=array((90,99,110)))
# print call(s=90, k=array((90,100,110)))

#PLot for v= 0, v = 0.1, etc
