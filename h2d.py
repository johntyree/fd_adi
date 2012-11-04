#!/usr/bin/env python

import scipy.weave
import numpy as np

I = 1j

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
        k_pi_recip_b_a = k*np.pi/(b-a)
        caba = (c-a) * k_pi_recip_b_a
        daba = (d-a) * k_pi_recip_b_a
        expd = np.exp(d)
        expc = np.exp(c)
        return (
            (
               (np.cos(daba) + k_pi_recip_b_a * np.sin(daba)) * expd
             - (np.cos(caba) + k_pi_recip_b_a * np.sin(caba)) * expc
            )
           / (1+(k_pi_recip_b_a)**2)
        )

    def psi(self, k, a, b, c, d):
        b_a = b - a
        kpi = k * np.pi
        k_pi_recip_b_a = kpi/(b_a)
        caba = (c-a) * k_pi_recip_b_a
        daba = (d-a) * k_pi_recip_b_a
        ret = (np.sin(daba) - np.sin(caba)) * b_a / kpi
        ret = np.dstack(((d-c), ret[:,:,1:]))
        return ret

    def CF(self, omega, r, var, T, kappa, theta, sigma, rho):
        p1 = kappa - 1j*rho*sigma*omega
        D = np.sqrt(p1**2 + (omega**2 + 1j*omega)*sigma**2)
        p0 = np.exp(-D*T)
        p1p = p1 + D
        p1 = p1 - D
        G = p1 / (p1p)
        p2 = 1-G*p0
        return (np.exp(1j*omega*r*T + var/sigma**2 * ((1-p0)/p2)*p1)
                * np.exp((kappa*theta)/sigma**2 * (T*p1 - 2*np.log(p2/(1-G)))))

    def COS(self, S, K, r, var, T, kappa, theta, sigma, rho):
        # Axes: [var, S, cosines]
        global U, a, b, U_tiled, CF_tiled, cf
        N = self.N
        L = 12
        x = np.log(S/K)[np.newaxis,:,np.newaxis]
        var = var[:,np.newaxis,np.newaxis]
        var_theta = var - theta
        c1 = r*T + (-var_theta)*((1 - np.exp(-kappa*T)*/(2*kappa)) - 0.5*theta*T

        p1 = (sigma*T*kappa*np.exp(-kappa*T)*(8*kappa*rho - 4*sigma)) * (var_theta)
        p2 = (kappa*rho*sigma*(1-np.exp(-kappa*T))*8)*(-var_theta + var)
        p3 = 2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)
        p4 = sigma**2*((-var_theta - var) * np.exp(-2*kappa*T) + theta*(6*np.exp(-kappa*T) - 7) + 2*var)
        p5 = (8*kappa**2*(1-np.exp(-kappa*T)))*(var_theta)
        c2 = 0.125*kappa**(-3) * (p1 + p2 + p3 + p4 + p5)


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


def bench():
    global ret
    spots = np.linspace(1,1200,200)
    vols  = np.linspace(0,10,200)
    ret = np.zeros((len(spots), len(vols)))
    h = HestonCos(spots, 100, 0.06, vols, 1, 1, 0.04, 0.2, 0)
    ret = h.solve()
