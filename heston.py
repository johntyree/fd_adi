#!/usr/bin/env python
"""Description."""

# import sys
# import os
import itertools as it
import time

import scipy.stats
import scipy.integrate
import scipy.weave
import numpy as np

from visualize import fp
prec = 3


I = 1j

fp = lambda x, y: None

BLACKSCHOLES, FUNDAMENTAL, COS = [2**i for i in range(3)]
HESTON = FUNDAMENTAL | COS
ALL = BLACKSCHOLES | HESTON

class HestonCos(object):
    def __init__(self, S, K, r, vol, T, kappa, theta, sigma, rho, N=2**8):
        r,T,kappa,theta,sigma,rho = map(float, [r,T,kappa, theta, sigma,rho])
        if hasattr(S, '__iter__') and hasattr(K, '__iter__'):
            raise TypeError("Can only have np.array(K) or np.array(S), not both.")
        elif hasattr(K, '__iter__'):
            K = np.array([K], copy=False, dtype=float)
            S = np.array([S], dtype=float)
        elif hasattr(S, '__iter__'):
            S = np.array(S, copy=False, dtype=float)
            K = np.array([K], dtype=float)
        else:
            S = np.array([S], dtype=float)
            K = np.array([K], dtype=float)

        if hasattr(vol, '__iter__'):
            vol = np.array(vol, dtype=float)
        else:
            vol = np.array((vol,), dtype=float)

        self.S     = S
        self.K     = K
        self.r     = r
        self.vol   = vol
        self.T     = T
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho   = rho
        self.N = 2**8

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
        ,"sigma: %s" % self.sigma
        ,"rho  : %s" % self.rho
        ,"     : %s" % self.N])



    def solve(self):
        ret = np.zeros_like(self.S)
        # ok = np.exp(np.log(self.S)) * 5*np.sqrt(self.vol) > self.K
        # ok = slice(None)
        # print "Cutoff:", min(self.S[ok]),
        ret = self.COS(self.S,
                        self.K,
                        self.r,
                        self.vol**2,
                        self.T,
                        self.kappa,
                        self.theta,
                        self.sigma,
                        self.rho,
                       )
        return ret


    def xi(self,k,a,b,c,d):
        return (
            1./(1+(k*np.pi/(b-a))**2) *
            (np.cos(k*np.pi*(d-a)/(b-a)) * np.exp(d)
             - np.cos(k*np.pi*(c-a)/(b-a)) * np.exp(c)
             + k*np.pi/(b-a) * np.sin(k*np.pi*(d-a)/(b-a)) * np.exp(d)
             - k*np.pi/(b-a) * np.sin(k*np.pi*(c-a)/(b-a)) * np.exp(c))
        )

    def psi(self, k, a, b, c, d):
        ret = (np.sin(k*np.pi*(d-a) / (b-a))
               - np.sin(k*np.pi*(c-a) / (b-a))
              ) * (b-a) / (k * np.pi)
        # print "psi ret", ret.shape
        # print "psi d-c", (d-c).shape
        ret = np.dstack(((d-c), ret[:,:,1:]))
        return ret

    def CF(self, omega, r, var, T, kappa, theta, sigma, rho):
        D = np.sqrt((kappa - 1j*rho*sigma*omega)**2 + (omega**2 + 1j*omega)*sigma**2)
        G = (kappa - 1j*rho*sigma*omega - D) / (kappa - 1j*rho*sigma*omega + D)
        # print "D", D.shape, fp(D, prec)
        # print "G", G.shape, fp(G, prec)
        return (np.exp(1j * omega * r * T + var/sigma**2 *
            ((1-np.exp(-D*T))/(1-G*np.exp(-D*T)))*(kappa - 1j*rho*sigma*omega - D))
            * np.exp((kappa*theta)/sigma**2 * (T*(kappa - 1j*rho*sigma*omega - D) -
                                          2*np.log((1-G*np.exp(-D*T))/(1-G)))))

    def COS(self, S, K, r, var, T, kappa, theta, sigma, rho):
        # Axes: [var, S, cosines]
        global U, a, b, U_tiled, CF_tiled, cf
        N = self.N
        L = 12
        x = np.log(S/K)[np.newaxis,:,np.newaxis]
        var = var[:,np.newaxis,np.newaxis]
        c1 = r*T + (1 - np.exp(-kappa*T))*(theta-var)/(2*kappa) - 0.5*theta*T
        c2 = 0.125*kappa**(-3) * (sigma*T*kappa*np.exp(-kappa*T)*(var-theta)*(8*kappa*rho - 4*sigma)
            + kappa*rho*sigma*(1-np.exp(-kappa*T))*(16*theta - 8*var)
            + 2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)
            + sigma**2*((theta - 2*var) * np.exp(-2*kappa*T) + theta*(6*np.exp(-kappa*T) - 7) + 2*var)
            + 8*kappa**2*(var-theta)*(1-np.exp(-kappa*T)))


        a = x + c1-L*np.sqrt(abs(c2))
        b = x + c1+L*np.sqrt(abs(c2))
        # print "a, b", a.shape
        k = np.arange(N)[np.newaxis,np.newaxis,:]
        # print "k", k.shape
        # print "var", var.shape

        NPOINTS = max(len(S), len(K))

        U = 2./(b-a)*(self.xi(k,a,b,0,b) - self.psi(k,a,b,0,b))
        # print "U", U.shape, fp(U,prec)
        # U_tiled = np.tile(U, (NPOINTS,1))
        # U_tiled = U
        # print "U_tiled", U_tiled.shape

        # print(S, K, r, var, T, kappa, theta, sigma, rho)

        cf = self.CF(k*np.pi/(b-a), r, var, T, kappa, theta, sigma, rho)
        cf[:,:,0] *= 0.5
        # print "CF", cf.shape, fp(cf, prec)
        ret = (cf * np.exp(1j*k*np.pi*(x-a)/(b-a))) * U
        # print "ret", ret.shape
        # print K * np.exp(-r*T) * ret.real
        ret = K * np.exp(-r*T) * ret.real.sum(axis=-1)
        ret[np.isnan(ret)] = 0
        return np.maximum(0, ret).T




# hc = HestonCos(100,99,0.06, 0.2, 1, 1, 0.2, 0.001, 0)


class HestonFundamental(object):

    def __init__(self, s, k, r, v, t, kappa, theta, sigma, rho):
        # if hasattr(s, '__iter__') and hasattr(k, '__iter__'):
            # raise TypeError("Can only have np.array(k) or np.array(s), not both.")
        # elif hasattr(k, '__iter__'):
            # k = np.array(k, copy=False)
        # elif hasattr(s, '__iter__'):
            # s = np.array(s, copy=False)
        # else:
            # k = np.array(k,)
        self.spot = s
        self.logspot = np.log(s)
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
            self.dmemo = np.sqrt(left-right)
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
             - 2.0 * np.log((1.0 - self.g(phi) * np.exp(self.d(phi) * self.tenor))
                     / (1.0 - self.g(phi)))))


    def D(self, phi):
        return ((self.b - self.rho * self.sig * phi * I + self.d(phi)) / pow(self.sig,2)
               * ((1.0 - np.exp(self.d(phi) * self.tenor))
                / (1.0 - self.g(phi) * np.exp(self.d(phi) * self.tenor))))


    def f(self, phi):
        return np.exp(self.C(phi) + self.D(phi)*self.var + I * phi * self.logspot)


    def integrand(self, phi):
        self.dmemo = self.gmemo = None
        return (np.exp(-I * phi * np.log(self.strike)) * self.f(phi) / (I * phi)).real

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
        # res.append(scipy.integrate.quad(self.integrand, 5e-2, inf)[0] * 1./np.pi + 0.5)
        res = self.ugly_integrate(self.integrand, 5e-2, 100, 1000)
        return res * 1./np.pi + 0.5

    def solve(self):
        # ok = np.exp(np.log(self.spot)) * 5*np.sqrt(np.sqrt(self.var)) > self.strike
        self.u = 0.5
        self.b = self.mean_reversion + self.lam - self.rho * self.sig
        P1 = self.P()

        self.u = -0.5
        self.b = self.mean_reversion + self.lam
        P2 = self.P()

        discount = np.exp(-self.r * self.tenor)

        return np.maximum(0, self.spot * P1 - self.strike * P2 * discount)

def hs_call(s, k, r, vol, t, kappa, theta, sigma, rho, HFUNC=HestonCos):
    h = HFUNC(s, k, r, vol, t, kappa, theta, sigma, rho).solve()
    # for j in range(len(vol)):
        # h.vol = vol[j]
        # ret[:,j] = h.solve()
    return h


def hs_stream(s, k, r, v, dt, kappa, theta, sigma, rho, HFUNC=HestonCos):
    ret = np.empty((len(s), len(v)))
    h = HFUNC(s, k, r, 0, 0, kappa, theta, sigma, rho)
    yield bs_call_delta(s[:,np.newaxis], k, r, v[np.newaxis,:], 0)[0]
    for i in it.count(1):
        yield hs_call(s, k, r, v, i*dt, kappa, theta, sigma, rho)
        # h.T = i*dt
        # for j in range(len(v)):
            # h.vol = v[j]
            # ret[:,j] = h.solve()
        # # print h
        # yield ret

def bs_call(s,k,r,v,t):
    return bs_call_delta(s,k,r,v,t)[0]

def bs_call_delta(s, k, r, vol, t):
    N = scipy.stats.distributions.norm.cdf
    d1 = (np.log(s/k) + (r+0.5*vol**2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol*np.sqrt(t)
    return (N(d1)*s - N(d2)*k*np.exp(-r * t), N(d1))

def call(s=100, k=99, r=0.06, vol=0.2, t=1, kappa=1, theta=None,
         sigma=1e-4, rho=0, method=COS, NCOS=2**8):
    if theta is None:
        theta = vol**2
    # s = np.array(s, dtype=float, copy=True)
    ret = []
    # return HestonSolver(s,k,r,vol,t,kappa,theta,sigma,rho).solve()
    if method & BLACKSCHOLES:
        try:
            x = np.array(bs_call_delta(s, k, r, vol, t))[:1]
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
        # try:
        x = HestonCos(s,k,r,vol,t,kappa,theta,sigma,rho,N=NCOS).solve()
        # except ValueError, e:
            # print "COS failed:", e
            # x = None
        # except TypeError, e:
            # print "COS failed:", e
            # x = None
        ret.append(x)
    return ret

# print call(s=np.array((90,100,110)), k=90)
# print call(s=np.array((90,100,110)), k=99, method=ALL)
# print call(s=np.array((90,100,110)), k=110)
# print call(s=110, k=np.array((90,100,110)))
# print call(s=100, k=np.array((90,99,110)))
# print call(s=90, k=np.array((90,100,110)))

#PLot for v= 0, v = 0.1, etc
