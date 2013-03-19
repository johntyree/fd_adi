#!/usr/bin/env python
"""Description."""


from __future__ import division

import sys
import os
import itertools as it
import time
from bisect import bisect_left

import scipy.stats
import scipy.integrate
import numpy as np
import numexpr as ne
import pylab

from Option import Option, BarrierOption, MeanRevertingProcess

from Grid import Grid
import utils

from visualize import fp
prec = 3

from FiniteDifferenceEngine import FiniteDifferenceEngineADI

from scipy.stats.distributions import norm
neval = ne.evaluate
norminv = norm.ppf
exp  = np.exp
log  = np.log
sqrt = np.sqrt

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
                , variance=None
                , tenor=1.0
                , mean_reversion=1
                , mean_variance=None
                , vol_of_variance=None
                , correlation=0):
        Option.__init__(self
                , spot=spot
                , strike=strike
                , interest_rate=interest_rate
                , variance=variance
                , volatility=volatility
                , tenor=tenor
                )
        self.Type = "HestonOption"
        self.attrs += ['correlation']
        self.variance.reversion = mean_reversion
        if mean_variance is not None:
            self.variance.mean = mean_variance
        else:
            self.variance.mean = self.variance.value
        if vol_of_variance is None:
            vol_of_variance = 0.4
        self.variance.volatility = vol_of_variance
        self.correlation = correlation

    def features(self):
        s = Option.features(self)
        s[0] = "HestonOption <%s>" % hex(id(self))
        s.extend([ "Mean Reversion: %s" % self.variance.reversion
                , "Mean Variance: %s" % self.variance.mean
                , "Vol of Variance: %s" % self.variance.volatility
                , "Correlation %s" % self.correlation
                ])
        return s


    def __repr__(self):
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs]
        s = self.Type + "(" + ", ".join(l) + ')'
        return s


    def compute_analytical(self):
        return HestonCos(
            self.spot,
            self.strike,
            self.interest_rate.value,
            self.volatility,
            self.tenor,
            self.variance.reversion,
            self.variance.mean,
            self.variance.volatility,
            self.correlation).solve()

    def monte_carlo_paths(self, dt=0.01, npaths=100000,
                          callback=lambda *x: None,
                          verbose=True):

        random_batch_size = 5
        neval = ne.evaluate
        norminv = norm.ppf
        exp  = np.exp
        log  = np.log
        sqrt = np.sqrt
        S0 = self.spot
        r = self.interest_rate.value
        V0 = self.variance.value
        t = self.tenor
        kpp = self.variance.reversion
        tht = self.variance.mean
        epp = self.variance.volatility
        rho_sv = self.correlation

        #parameter values used for discretization, =0.5 uses central discretization
        gmm1 = 0.5;
        gmm2 = 0.5;
        #constant for the switching rule used in the QE scheme
        shi_crt = 1.5;

        nrOfSteps = int(t/dt)

        Vt = np.empty((npaths,))
        St = np.empty((npaths,))
        Vt[:] = V0
        St[:] = S0

        # Whether or not this path will count
        state = np.ones(npaths, dtype=bool)

        # U_all = np.random.random((nrOfSteps, npaths))
        # Z1_all = norminv(U_all)
        # Z2_all = np.random.standard_normal((nrOfSteps, npaths))

        notify = nrOfSteps // 10
        notify += notify == 0
        for i in range(nrOfSteps):
            if verbose:
                if not i % notify:
                    print int(100*i / nrOfSteps),
                    sys.stdout.flush()
            V = Vt

            #Andersen's paper, equation (17)
            m = neval("tht + (V-tht)*exp(-kpp*dt)")
            #Andersen's paper, equation (18)
            s2 = neval("V*epp**2*exp(-kpp*dt)*(1-exp(-kpp*dt))/kpp + tht*epp**2*(1-exp(-kpp*dt))**2/(2*kpp)")
            # s2 = V * (epp**2*exp(-kpp*dt)*(1-exp(-kpp*dt))/kpp) + (tht*epp**2*(1-exp(-kpp*dt))**2/(2*kpp))
            #Andersen's paper, equation (19)
            shi = neval("s2/(m**2)")
            #Andersen's paper, p19, where C0 - K0, C1 - K1, C2 - K2, C3 - sqrt(K3/gmm1)
            C0 = (-rho_sv*tht*kpp/epp) * dt;
            C1 = gmm1*dt*(kpp*rho_sv/epp-0.5) - rho_sv/epp;
            C2 = gmm2*dt*(kpp*rho_sv/epp-0.5) + rho_sv/epp;
            C3 = sqrt((1-rho_sv**2)*dt);

            #Andersen's paper, p20, A - AVtdt
            AVtdt = C2 + 0.5*C3**2*gmm2;
            # assert not np.isnan(AVtdt).any()
            #Andersen's QE algorithm 3.2.4, p16 - 17
            if not i % random_batch_size:
                U = np.random.random((random_batch_size, npaths,))
                Z = np.random.standard_normal((random_batch_size*2, npaths))
            Z0 = Z[i % random_batch_size]
            Z1 = Z[i % random_batch_size + random_batch_size]
            u = U[i % random_batch_size]

            boolvec = shi <= shi_crt
            #for sufficiently large value s of Vt
            #condition to be satisfied for QE martingale correction: AVtdt < (1/(2*a))
            c4 = 2/shi
            b2 = np.maximum(c4-1+sqrt(c4*(c4-1)),0)
            # assert not np.isnan(b2).any()
            a = m/(1+b2)
            # assert not np.isnan(a).any()
            Vdt = neval("(a*(sqrt(b2)+Z1)**2) * boolvec")
            # assert not np.isnan(Vdt).any()
            #Martingale drift correction, p22, K0_star -- C00
            C00 = neval("(-AVtdt*b2*a/(1-2*AVtdt*a) + 0.5*log(1-2*AVtdt*a) - (C1+0.5*C3**2*gmm1)*V)*boolvec")
            # assert not np.isnan(C00).any()

            #for low values of Vt
            #condition to be satisfied for QE martingale correction: AVtdt < beta
            p = (shi-1)/(shi+1);
            # assert not np.isnan(p).any()
            bet = (1-p)/m;
            # assert not np.isnan(bet).any()
            tmpVdt = neval("log((1-p)/(1-u))/bet * (u > p)")
            # assert not np.isnan(tmpVdt).any()
            #Martingale drift correction, p22, K0_star -- C00
            tmpC00 = neval("-log(p + bet*(1-p)/(bet-AVtdt)) - (C1+0.5*C3**2*gmm1)*V")
            # assert not np.isnan(tmpC00).any()

            Vdt[-boolvec] = tmpVdt[-boolvec]
            C00[-boolvec] = tmpC00[-boolvec]
            # assert not np.isnan(Vdt).any()
            # assert not np.isnan(C00).any()

            #simulated Heston stock prices
            #Andersen's paper, p19, equation (33), with drift corrected, K0 replaced by C00
            St = neval("St * exp(r*dt + C00 + C1*V + C2*Vdt + C3*sqrt(gmm1*V+gmm2*Vdt)*Z0)")
            callback(St, state)
            # assert not np.isnan(St[i,:]).any()

            #update Heston stochastic variance
            Vt = Vdt
            # assert not np.isnan(Vt[i+1,:]).any()

        return St * state



class HestonBarrierOption(HestonOption, BarrierOption):
    """#:class documentation"""
    def __init__(self
            , spot=100
            , strike=99
            , interest_rate=0.06
            , volatility=0.2
            , variance=None
            , tenor=1.0
            , mean_reversion=1
            , mean_variance=None
            , vol_of_variance=0.4
            , correlation=0
            , top=None
            , bottom=None
            ):
        """"""
        # We must call Barrier BEFORE Heston or our variance process will be
        # ruined, but if we do... monte_carlo_callback is ruined...
        BarrierOption.__init__(self, spot=spot, strike=strike,
                interest_rate=interest_rate,
                volatility=volatility, variance=variance,
                tenor=tenor, top=top, bottom=bottom)
        HestonOption.__init__(self
                , spot=spot
                , strike=strike
                , interest_rate=interest_rate
                , volatility=volatility
                , tenor=tenor
                , mean_reversion=mean_reversion
                , mean_variance=mean_variance
                , vol_of_variance=vol_of_variance
                , correlation=correlation
                )
        self.Type = 'HestonBarrierOption'


    def __repr__(self):
        l = ["{attr}={val}".format(attr=attr, val=repr(getattr(self, attr))) for attr in self.attrs]
        s = self.Type + "(" + ", ".join(l) + ')'
        return s


    def compute_analytical(self):
        raise NotImplementedError("No analytical solution for Heston barrier options.")


    def features(self):
        d = HestonOption.features(self)
        d[0] = "HestonBarrierOption <%s>" % hex(id(self))
        b = BarrierOption.features(self)
        d.extend(b[-2:])
        return d


class HestonFiniteDifferenceEngine(FiniteDifferenceEngineADI):
    """FDE specialized for Heston options."""
    def __init__(self, option,
            grid=None,
            spot_max=1500.0,
            spot_min=0.0,
            spots=None,
            vars=None,
            var_max=10.0,
            nspots=100,
            nvols=100,
            spotdensity=7.0,
            varexp=4.0,
            force_exact=True,
            flip_idx_var=False,
            flip_idx_spot=False,
            schemes=None,
            coefficients=None,
            boundaries=None,
            cache=True,
            verbose=True,
            force_bandwidth=None
            ):
        """@option@ is a HestonOption"""
        self.cache = cache
        assert isinstance(option, Option)
        self.option = option

        if not coefficients:
            def mu_s(t, *dim):
                # return option.interest_rate.value - 0.5 * dim[1]
                return option.interest_rate.value * dim[0]
            def gamma2_s(t, *dim):
                # return 0.5 * dim[1]
                return 0.5 * dim[1] * dim[0]**2
            def mu_v(t, *dim):
                if np.isscalar(dim[0]):
                    if dim[0] == 0:
                        return 0
                ret = option.variance.reversion * (option.variance.mean - dim[1])
                ret[dim[0]==0] = 0
                return ret
            def gamma2_v(t, *dim):
                if np.isscalar(dim[0]):
                    if dim[0] == 0:
                        return 0
                ret = 0.5 * option.variance.volatility**2 * dim[1]
                ret[dim[0]==0] = 0
                return ret
            def cross(t, *dim):
                # return option.correlation * option.variance.volatility * dim[1]
                return option.correlation * option.variance.volatility * dim[0] * dim[1]

            coefficients = {()   : lambda t: -option.interest_rate.value,
                    (0,) : mu_s,
                    (0,0): gamma2_s,
                    (1,) : mu_v,
                    (1,1): gamma2_v,
                    (0,1): cross,
                    }

        if not boundaries:
            boundaries = {
                            # D: U = 0              VN: dU/dS = 1
                    # (0,)  : ((0, lambda t, *dim: 0.0), (1, lambda t, *dim: np.exp(dim[0]))),
                    (0,)  : ((0, lambda t, *dim: 0.0), (1, lambda t, *dim: 1.0)),
                            # D: U = 0              Free boundary
                    # (0,0) : ((0, lambda t, *dim: 0.0), (None, lambda t, *dim:  np.exp(dim[0]))),
                    (0,0) : ((0, lambda t, *dim: 0.0), (None, lambda t, *dim: 1.0)),
                            # Free boundary at low variance
                    (1,)  : ((None, lambda t, *dim: None),
                            # # D intrinsic value at high variance
                            # (0, lambda t, *dim: np.exp(-option.interest_rate.value * t) * dim[0])
                            (None, lambda t, *dim: None)
                            # (0, lambda t, *dim: dim[0])
                            ),
                            # We know from the PDE that this will be 0 because
                            # the vol is 0 at the low boundary
                    (1,1) : ((1, lambda t, *dim: 0),
                            # D intrinsic value at high variance
                            # (0, lambda t, *dim: np.exp(-option.interest_rate.value * t) * np.maximum(0.0, np.exp(dim[0])-option.strike))),
                            (None, lambda t, *dim: None)
                            # (0, lambda t, *dim: dim[0])
                            # (0, lambda t, *dim: 0)
                            )
                    }

        if isinstance(option, BarrierOption):
            if option.top:
                if option.top[0]: # Knockin, not sure about implementing this
                    raise NotImplementedError("Knockin barriers are not supported.")
                else:
                    spot_max = option.top[1]
                    if grid:
                        assert np.allclose(spot_max, max(grid.mesh[0]))
                    boundaries[(0,)] = (boundaries[(0,)][0], (0, lambda *x: 0.0))
                    boundaries[(0,0)] = boundaries[(0,)]
            if option.bottom:
                if option.bottom[0]: # Knockin, not sure about implementing this
                    raise NotImplementedError("Knockin barriers are not supported.")
                else:
                    spot_min = option.bottom[1]
                    boundaries[(0,)] = ((0, lambda *x: 0.0), boundaries[(0,)][1])
                    boundaries[(0,0)] = boundaries[(0,)]


        if grid:
            self.spots = grid.mesh[0]
            self.vars = grid.mesh[1]
        else:
            if vars is None:
                # vars = np.linspace(0, var_max, nvols)
                vars = utils.exponential_space(0.00, option.variance.value, var_max,
                                            varexp, nvols,
                                            force_exact=force_exact)
            self.vars = vars
            if spots is None:
                # spots = np.linspace(0,spot_max,nspots)
                if isinstance(option, BarrierOption) and option.top and not option.top[0]:
                        p = 3
                        spots = np.linspace(0, spot_max**p, nspots)**(1.0/p)
                        print "Barrier spots"
                else:
                    spots = utils.sinh_space(option.strike-spot_min, spot_max-spot_min, spotdensity, nspots, force_exact=force_exact) + spot_min
            self.spots = spots
            grid = Grid([self.spots, self.vars], initializer=lambda *x: np.maximum(x[0]-option.strike,0))


        newstrike = self.spots[np.argmin(np.abs(self.spots - option.strike))]
        self.spots[np.argmin(np.abs(self.spots - option.spot))] = option.spot
        # if newstrike != option.strike:
            # print "Strike %s -> %s" % (option.strike, newstrike)
            # option.strike = newstrike
        # if newspot != option.spot:
            # print "Spot %s -> %s" % (option.spot, newspot)
            # option.spot = newspot

        if flip_idx_var is True: # Need explicit boolean True
            flip_idx_var = bisect_left(
                    np.round(self.vars, decimals=5),
                    np.round(option.variance.mean, decimals=5))
        if flip_idx_spot is True: # Need explicit boolean True
            flip_idx_spot = bisect_left(
                    np.round(self.spots, decimals=5),
                    np.round(option.strike, decimals=5))


        if schemes is None:
            schemes = {}
        else:
            schemes = {k : list(v) for k, v in schemes.items()}
            # for k,v in new.items():
                # assert schemes[k] is not v
            # schemes = new
        if (0,) not in schemes:
            schemes[(0,)] = [{"scheme": "center"}]
        if flip_idx_spot is not False:
            schemes[(0,)].append({"scheme": 'forward', "from" : flip_idx_spot})

        if (1,) not in schemes:
            schemes[(1,)] = [{"scheme": "center"}]
        if flip_idx_var is not False:
            schemes[(1,)].append({"scheme": 'backward', "from" : flip_idx_var})

        if verbose:
            print "(0,): Start with %s differencing." % (schemes[(0,)][0]['scheme'],)
            if len(schemes[(0,)]) > 1:
                print "(0,): Switch to %s differencing at %i." % (schemes[(0,)][1]['scheme'], schemes[(0,)][1]['from'])
            print "(1,): Start with %s differencing." % (schemes[(1,)][0]['scheme'],)
            if len(schemes[(1,)]) > 1:
                print "(1,): Switch to %s differencing at %i." % (schemes[(1,)][1]['scheme'], schemes[(1,)][1]['from'])


        self.grid = grid
        self.coefficients = coefficients
        self.boundaries = boundaries
        self.schemes = schemes
        self.force_bandwidth = force_bandwidth
        self._initialized = False

        # FiniteDifferenceEngineADI.__init__(self, G, coefficients=self.coefficients,
                # boundaries=self.boundaries, schemes=self.schemes, force_bandwidth=(-2,2))


    @property
    def idx(self):
        ids = bisect_left(np.round(self.spots, decimals=4), np.round(self.option.spot, decimals=4))
        idv = bisect_left(np.round(self.vars, decimals=4), np.round(self.option.variance.value, decimals=4))
        return (ids, idv)

    @property
    def price(self):
        return self.grid.domain[-1][self.idx]


    @property
    def grid_analytical(self):
        H = self.option
        if isinstance(H, BarrierOption):
            raise NotImplementedError("No analytical solution for Heston barrier options.")
        hs = hs_call_vector(self.spots, H.strike,
            H.interest_rate.value, np.sqrt(self.vars), H.tenor,
            H.variance.reversion, H.variance.mean, H.variance.volatility,
            H.correlation, HFUNC=HestonCos, cache=self.cache)

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
                # print "Loading:", fname
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
        ret = ne.evaluate("""(
            ((cos(daba) + k_pi_recip_b_a * sin(daba)) * expd
            - (cos(caba) + k_pi_recip_b_a * sin(caba)) * expc)
            / (1+(k_pi_recip_b_a)**2))""")
        del caba, daba, expd, expc, k_pi_recip_b_a
        return ret

    def psi(self, k, a, b, c, d):
        sin = np.sin
        b_a = b - a
        kpi = k * np.pi
        k_pi_recip_b_a = kpi/(b_a)
        caba = (c-a) * k_pi_recip_b_a
        daba = (d-a) * k_pi_recip_b_a
        ret = ne.evaluate("(sin(daba) - sin(caba)) * b_a / kpi")
        del caba, daba, b_a, k_pi_recip_b_a, kpi
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
        del p10
        G = p1m / p1p
        del p1p
        p0 = ne.evaluate("exp(-D*T)")
        p2 = 1-G*p0
        ret = ne.evaluate("""(
              exp(I*omega*r*T + var/sigma**2 * ((1-p0)/p2)*p1m)
            * exp(  (kappa*theta)/sigma**2
                  * (T*p1m - 2*log(p2/(1-G)))))""")
        del D, G, p0, p2
        return ret

    def COS(self, S, K, r, var, T, kappa, theta, sigma, rho):
        # Axes: [var, S, cosines]
        global U, a, b, U_tiled, CF_tiled, cf
        N = self.N
        L = 12
        x = np.log(S/K)[np.newaxis,:,np.newaxis]
        var = var[:,np.newaxis,np.newaxis]
        var_theta = var - theta

        p1 = (sigma*T*kappa*np.exp(-kappa*T)*(8*kappa*rho - 4*sigma)) * (var_theta)
        p2 = (kappa*rho*sigma*(1-np.exp(-kappa*T))*8)*(-var_theta + var)
        p3 = 2*theta*kappa*T*(-4*kappa*rho*sigma+sigma**2+4*kappa**2)
        p4 = sigma**2*((-var_theta - var) * np.exp(-2*kappa*T) + theta*(6*np.exp(-kappa*T) - 7) + 2*var)
        p5 = (8*kappa**2*(1-np.exp(-kappa*T)))*(var_theta)
        c2 = ne.evaluate("0.125*kappa**(-3) * (p1 + p2 + p3 + p4 + p5)")
        del p1, p2, p3, p4, p5

        # print (S, K, r, var, T, kappa, theta, sigma, rho)
        # print "x", x
        # print "p1", p1
        # print "p2", p2
        # print "p3", p3
        # print "p4", p4
        # print "p5", p5

        c1 = r*T + (-var_theta)*(1 - np.exp(-kappa*T))/(2*kappa) - 0.5*theta*T
        Lc2 = L*np.sqrt(abs(c2))
        del c2
        a = x + c1-Lc2
        b = x + c1+Lc2
        del Lc2, c1
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
        del XI, PSI

        cf = self.CF(k*np.pi/(b-a), r, var, T, kappa, theta, sigma, rho)
        cf[:,:,0] *= 0.5
        # print "cf:", cf
        pi = np.pi
        ret = ne.evaluate("(cf * exp(I*k*pi*(x-a)/(b-a))) * U").real
        del cf

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

    def __str__(self):
        return '\n'.join([
        "Heston Fundamental object: %s" % id(self), self.param_str()])

    def param_str(self):
        return '\n'.join([
         "s    : %s" % self.spot
        ,"k    : %s" % self.strike
        ,"r    : %s" % self.r
        ,"vol  : %s" % np.sqrt(self.var)
        ,"t    : %s" % self.tenor
        ,"kappa: %s" % self.mean_reversion
        ,"theta: %s" % self.mean_variance
        ,"sigma: %s" % self.sig
        ,"rho  : %s" % self.rho
        ])


    def solve(self, cache=True):
        cache_dir = 'heston_cache'
        fname = os.path.join(cache_dir, "FUNDAMENTAL_"+str(hash(self.param_str()))+".npy")
        if cache:
            # print "Seeking:", fname
            if os.path.isfile(fname):
                # print "Loading:", fname
                return np.load(fname)
        cols = []
        vars = np.atleast_1d(self.var)
        for v in vars:
            self.var = v
            self.u = 0.5
            self.b = self.mean_reversion + self.lam - self.rho * self.sig
            P1 = self.P()

            self.u = -0.5
            self.b = self.mean_reversion + self.lam
            P2 = self.P()

            discount = np.exp(-self.r * self.tenor)

            cols.append(np.maximum(0, self.spot * P1 - self.strike * P2 * discount))
        self.var = vars
        ret = np.array(cols).T
        if cache:
            if not os.path.isdir(cache_dir):
                os.mkdir(cache_dir)
            np.save(fname, ret)
        return ret


def hs_call_vector(s, k, r, vols, t, kappa, theta, sigma, rho, HFUNC=HestonCos, cache=True):
    # if s[0] == 0:
        # ret0 = HFUNC(s[1:], k, r, vols, t, kappa, theta, sigma, rho).solve(cache=cache)
        # ret0 = np.vstack((np.zeros((1, len(vols))), ret))
    # else:
        # ret0 = HFUNC(s, k, r, vols, t, kappa, theta, sigma, rho).solve(cache=cache)
    ret = np.empty((len(s), len(vols)))
    if s[0] == 0:
        ret[0,:] = 0.0
        for i in range(0, len(vols), 128):
            ret[1:, i:i+128] = HFUNC(s[1:], k, r, vols[i:i+128], t, kappa, theta, sigma, rho).solve(cache=cache)
    else:
        for i in range(0, len(vols), 128):
            ret[:, i:i+128] = HFUNC(s, k, r, vols[i:i+128], t, kappa, theta, sigma, rho).solve(cache=cache)
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
