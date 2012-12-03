#!/usr/bin/env python
"""Description."""

# import sys
import os
import itertools as it
import time
from bisect import bisect_left

import scipy.stats
import scipy.integrate
import numpy as np
import numexpr as ne
import pylab

from Option import Option

from FiniteDifferenceEngine import FiniteDifferenceEngineADI
from Grid import Grid
import utils

from visualize import fp
prec = 3


I = 1j

BLACKSCHOLES, FUNDAMENTAL, COS = [2**i for i in range(3)]
HESTON = FUNDAMENTAL | COS
ALL = BLACKSCHOLES | HESTON

class HestonOption(Option):
    def __init__(self
                , spot=100
                , strike=99
                , interest_rate=0.06
                , volatility=0.2
                , tenor=1.0
                , mean_reversion=1
                , mean_variance=None
                , vol_of_variance=0.4
                , correlation=0):
        Option.__init__(self
                , spot=spot
                , strike=strike
                , interest_rate=interest_rate
                , volatility=volatility
                , tenor=tenor
                )
        self.mean_reversion = mean_reversion
        self.mean_variance = mean_variance if mean_variance is not None else volatility**2
        self.vol_of_variance = vol_of_variance
        self.correlation = correlation

    def __str__(self):
        s = Option.features(self)
        s[0] = "HestonOption <%s>" % hex(id(self))
        s.extend(
                [ "Mean Reversion: %s" % self.mean_reversion
                , "Mean Variance: %s" % self.mean_variance
                , "Vol of Variance: %s" % self.vol_of_variance
                , "Correlation %s" % self.correlation
                ])
        return s

    @property
    def analytical(self):
        return HestonCos(
            self.spot,
            self.strike,
            self.interest_rate.value,
            self.volatility,
            self.tenor,
            self.mean_reversion,
            self.mean_variance,
            self.vol_of_variance,
            self.correlation).solve()


class HestonFiniteDifferenceEngine(FiniteDifferenceEngineADI):
    """FDE specialized for Heston options."""
    def __init__(self, option,
            spot_max=1500.0,
            var_max=10.0,
            nspots=100,
            nvols=100,
            spotdensity=7.0,
            varexp=4.0,
            force_exact=True,
            flip_idx_var=True,
            flip_idx_spot=False,
            schemes=None
            ):
        """@option@ is a HestonOption"""

        H = option
        self.option = H

        spots = utils.sinh_space(H.strike, spot_max, spotdensity, nspots, force_exact=force_exact)
        # spots = (2.0 * np.arange(nspots)) / nspots  * np.log(H.strike)
        # vars = utils.exponential_space(0.00, H.variance.value, var_max, varexp, nvols)
        # vars = [v0]
        # spots = np.linspace(0.0, spot_max, nspots)
        vars = np.linspace(0.0, var_max, nvols)
        # plot(spots); title("Spots"); show()
        # plot(vars); title("Vars"); show()
        self.spots = spots
        # self.spots = np.exp(spots)
        print self.spots, spots
        self.vars = vars

        H.strike = spots[min(abs(self.spots - H.strike)) == abs(self.spots - H.strike)][0]
        H.spot = spots[min(abs(self.spots - H.spot)) == abs(self.spots - H.spot)][0]

        # def init(logspots, vars):
            # return np.maximum(0, np.exp(logspots) - H.strike)
        def init(spots, vars):
            return np.maximum(0, spots - H.strike)
        G = Grid(mesh=(spots, vars), initializer=init)

        if flip_idx_var is True:
            flip_idx_var = bisect_left(
                    np.round(vars, decimals=5),
                    np.round(H.mean_variance, decimals=5))
        if flip_idx_spot is True:
            flip_idx_spot = bisect_left(
                    np.round(spots, decimals=5),
                    np.round(H.strike, decimals=5))


        def mu_s(t, *dim):
            # return H.interest_rate.value - 0.5 * dim[1]
            return H.interest_rate.value * dim[0]
        def gamma2_s(t, *dim):
            # return 0.5 * dim[1]
            return 0.5 * dim[1] * dim[0]**2
        def mu_v(t, *dim):
            if dim[0] == 0:
                ret = 0
            else:
                ret = H.mean_reversion * (H.mean_variance - dim[1])
            return ret
        def gamma2_v(t, *dim):
            if dim[0] == 0:
                ret = 0
            else:
                ret = 0.5 * H.vol_of_variance**2 * dim[1]
            return ret
        def cross(t, *dim):
            # return H.correlation * H.vol_of_variance * dim[1]
            return H.correlation * H.vol_of_variance * dim[0] * dim[1]

        self.coefficients = {()   : lambda t: -H.interest_rate.value,
                  (0,) : mu_s,
                  (0,0): gamma2_s,
                  (1,) : mu_v,
                  (1,1): gamma2_v,
                  (0,1): cross,
                  }

        self.boundaries = {
                        # D: U = 0              VN: dU/dS = 1
                # (0,)  : ((0, lambda t, *dim: 0.0), (1, lambda t, *dim: np.exp(dim[0]))),
                (0,)  : ((0, lambda t, *dim: 0.0), (1, lambda t, *dim: 1.0)),
                        # D: U = 0              Free boundary
                # (0,0) : ((0, lambda t, *dim: 0.0), (None, lambda t, *dim:  np.exp(dim[0]))),
                (0,0) : ((0, lambda t, *dim: 0.0), (None, lambda t, *dim: 1.0)),
                        # Free boundary at low variance
                (1,)  : ((None, lambda t, *dim: None),
                        # # D intrinsic value at high variance
                        # (0, lambda t, *dim: np.exp(-H.interest_rate.value * t) * np.maximum(0.0, np.exp(dim[0])-H.strike))),
                        # (0, lambda t, *dim: np.maximum(0.0, dim[0]-H.strike))),
                        (0, lambda t, *dim: 0)),
                        # # Free boundary
                (1,1) : ((None, lambda t, *dim: None),
                        # D intrinsic value at high variance
                        # (0, lambda t, *dim: np.exp(-H.interest_rate.value * t) * np.maximum(0.0, np.exp(dim[0])-H.strike))),
                        # (0, lambda t, *dim: np.maximum(0.0, dim[0]-H.strike)))
                        (0, lambda t, *dim: 0)),
                }

        if schemes is None:
            schemes = {}
        if (0,) not in schemes:
            schemes[(0,)] = [{"scheme": "center"}]
        print "(0,): Start with %s differencing." % (schemes[(0,)][0]['scheme'],)
        if flip_idx_spot is not False:
            schemes[(0,)].append({"scheme": 'forward', "from" : flip_idx_spot})
        if len(schemes[(0,)]) > 1:
            print "(0,): Switch to %s differencing at %i." % (schemes[(0,)][1]['scheme'], schemes[(0,)][1]['from'])

        if (1,) not in schemes:
            schemes[(1,)] = [{"scheme": "center"}]
        print "(1,): Start with %s differencing." % (schemes[(1,)][0]['scheme'],)
        if flip_idx_var is not False:
            schemes[(1,)].append({"scheme": 'backward', "from" : flip_idx_var})
        if len(schemes[(1,)]) > 1:
            print "(1,): Switch to %s differencing at %i." % (schemes[(1,)][1]['scheme'], schemes[(1,)][1]['from'])
        self.schemes = schemes

        FiniteDifferenceEngineADI.__init__(self, G, coefficients=self.coefficients,
                boundaries=self.boundaries, schemes=self.schemes, force_bandwidth=(-2,2))

        ids = bisect_left(np.round(self.spots, decimals=4), np.round(self.option.spot, decimals=4))
        idv = bisect_left(np.round(self.vars, decimals=4), np.round(self.option.variance.value, decimals=4))
        self.idx = (ids, idv)

    @property
    def price(self):
        return self.grid.domain[-1][self.idx]


    @property
    def grid_analytical(self):
        H = self.option
        hs = hs_call_vector(self.spots, H.strike,
            H.interest_rate.value, np.sqrt(self.vars), H.tenor,
            H.mean_reversion, H.mean_variance, H.vol_of_variance,
            H.correlation)

        if max(hs.flat) > self.spots[-1] * 2:
            self.BADANALYTICAL = True
            print "Warning: Analytical solution looks like trash."
        else:
            self.BADANALYTICAL = False
        return hs






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
        self.N = N

    def __str__(self):
        return '\n'.join([
        "Heston cos object: %s" % id(self), self.param_str()])

    def param_str(self):
        return '\n'.join([
         "s    : %s" % self.S
        ,"k    : %s" % self.K
        ,"r    : %s" % self.r
        ,"vol  : %s" % self.vol
        ,"t    : %s" % self.T
        ,"kappa: %s" % self.kappa
        ,"theta: %s" % self.theta
        ,"sigma: %s" % self.sigma
        ,"rho  : %s" % self.rho
        ,"N    : %s" % self.N])



    def solve(self, cache=True):
        cache_dir = 'heston_cache'
        fname = os.path.join(cache_dir, "x"+str(hash(self.param_str()))+".npy")
        if cache:
            # print "Seeking:", fname
            if os.path.isfile(fname):
                print "Loading:", fname
                return np.load(fname)

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
        # Value should be monotonic w.r.t. underlying and volatility. If not, we
        # assume the method is breaking down and correct for it.
        # print "Setting price to 0 for spot <", self.S[flattenbelow]
        if ret.shape[0] != 1:
            diffs = np.diff(ret, axis=0)
            mask = np.vstack((diffs[0, :], diffs)) <= 0
            ret = np.where(mask, 0, ret)
        # del diffs, mask
        # diffs2 = np.diff(ret, n=2, axis=1)
        # diffs2 = np.hstack((diffs2[:, :1], diffs2[:, :1], diffs2))
        # mask = diffs2 / self.S > 0.1
        # diffs = np.cumsum(np.where(mask, 0, diffs2), axis=1)
        # mask = diffs / self.S < -0.1
        # ret = np.cumsum(np.where(mask, 0, diffs), axis=1)
        if cache:
            if not os.path.isdir(cache_dir):
                os.mkdir(cache_dir)
            np.save(fname, ret)
        return ret


    def xi(self,k,a,b,c,d):
        pi = np.pi
        cos = np.cos
        sin = np.sin
        exp = np.exp
        k_pi_recip_b_a = ne.evaluate("k*pi/(b-a)")
        caba = (c-a) * k_pi_recip_b_a
        daba = (d-a) * k_pi_recip_b_a
        expd = ne.evaluate("exp(d)")
        expc = ne.evaluate("exp(c)")
        return ne.evaluate("""(
            ((cos(daba) + k_pi_recip_b_a * sin(daba)) * expd
            - (cos(caba) + k_pi_recip_b_a * sin(caba)) * expc)
            / (1+(k_pi_recip_b_a)**2))""")

    def psi(self, k, a, b, c, d):
        sin = np.sin
        b_a = b - a
        kpi = k * np.pi
        k_pi_recip_b_a = kpi/(b_a)
        caba = (c-a) * k_pi_recip_b_a
        daba = (d-a) * k_pi_recip_b_a
        ret = ne.evaluate("(sin(daba) - sin(caba)) * b_a / kpi")
        ret = np.dstack(((d-c), ret[:,:,1:]))
        return ret

    def CF(self, omega, r, var, T, kappa, theta, sigma, rho):
        pi = np.pi
        cos = np.cos
        sin = np.sin
        exp = np.exp
        sqrt = np.sqrt
        log = np.log
        p10 = ne.evaluate("kappa - (1j*rho*sigma)*omega")
        D = ne.evaluate("sqrt(p10**2 + (omega**2 + 1j*omega)*sigma**2)")
        p1p = p10 + D
        p1m = p10 - D
        G = p1m / p1p
        p0 = ne.evaluate("exp(-D*T)")
        p2 = 1-G*p0
        return ne.evaluate("""(
              exp(I*omega*r*T + var/sigma**2 * ((1-p0)/p2)*p1m)
            * exp(  (kappa*theta)/sigma**2
                  * (T*p1m - 2*log(p2/(1-G)))))""")

    def COS(self, S, K, r, var, T, kappa, theta, sigma, rho):
        # Axes: [var, S, cosines]
        global U, a, b, U_tiled, CF_tiled, cf
        N = self.N
        L = 12
        x = np.log(S/K)[np.newaxis,:,np.newaxis]
        var = var[:,np.newaxis,np.newaxis]
        var_theta = var - theta
        c1 = r*T + (-var_theta)*(1 - np.exp(-kappa*T))/(2*kappa) - 0.5*theta*T

        p1 = (sigma*T*kappa*np.exp(-kappa*T)*(8*kappa*rho - 4*sigma)) * (var_theta)
        p2 = (kappa*rho*sigma*(1-np.exp(-kappa*T))*8)*(-var_theta + var)
        p3 = 2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)
        p4 = sigma**2*((-var_theta - var) * np.exp(-2*kappa*T) + theta*(6*np.exp(-kappa*T) - 7) + 2*var)
        p5 = (8*kappa**2*(1-np.exp(-kappa*T)))*(var_theta)
        c2 = ne.evaluate("0.125*kappa**(-3) * (p1 + p2 + p3 + p4 + p5)")

        # print (S, K, r, var, T, kappa, theta, sigma, rho)
        # print "x", x
        # print "p1", p1
        # print "p2", p2
        # print "p3", p3
        # print "p4", p4
        # print "p5", p5

        a = x + c1-L*np.sqrt(abs(c2))
        b = x + c1+L*np.sqrt(abs(c2))
        k = np.arange(N)[np.newaxis,np.newaxis,:]

        # print "a", a
        # print "b", b
        # print "c1", c1
        # print "c2", c2

        NPOINTS = max(len(S), len(K))

        XI = self.xi(k,a,b,0,b)
        # print "xi:", XI
        PSI = self.psi(k,a,b,0,b)
        # print "psi:", PSI
        U = ne.evaluate("2./(b-a)*(XI - PSI)")

        cf = self.CF(k*np.pi/(b-a), r, var, T, kappa, theta, sigma, rho)
        cf[:,:,0] *= 0.5
        # print "cf:", cf
        pi = np.pi
        ret = ne.evaluate("(cf * exp(I*k*pi*(x-a)/(b-a))) * U").real

        # print "ret:", ret
        ret = K * np.exp(-r*T) * ret.real.sum(axis=-1)
        ret[np.isnan(ret)] = 0
        return np.maximum(0, ret).T


class HestonFundamental(object):
    def __init__(self, s, k, r, v, t, kappa, theta, sigma, rho):
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

def hs_call_vector(s, k, r, vols, t, kappa, theta, sigma, rho, HFUNC=HestonCos):
    if s[0] == 0:
        ret = HFUNC(s[1:], k, r, vols, t, kappa, theta, sigma, rho).solve()
        ret = np.vstack((np.zeros((1, len(vols))), ret))
    else:
        ret = HFUNC(s, k, r, vols, t, kappa, theta, sigma, rho).solve()
    return ret

def hs_stream(s, k, r, v, dt, kappa, theta, sigma, rho, HFUNC=HestonCos):
    ret = np.empty((len(s), len(v)))
    h = HFUNC(s, k, r, 0, 0, kappa, theta, sigma, rho)
    yield bs_call_delta(s[:,np.newaxis], k, r, v[np.newaxis,:], 0)[0]
    for i in it.count(1):
        yield hs_call(s, k, r, v, i*dt, kappa, theta, sigma, rho)

def bench():
    global ret
    spots = np.linspace(1,1200,200)
    vols  = np.linspace(0,10,200)
    ret = np.zeros((len(spots), len(vols)))
    h = HestonCos(spots, 100, 0.06, vols, 1, 1, 0.04, 0.2, 0)
    ret = h.solve()

def main():
    print HestonCos((100,),99,0.06,(0.0,), 1, 1, 0.04, 0.001, 0).solve(cache=False)


if __name__ == "__main__":
    main()
