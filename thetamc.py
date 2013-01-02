#this model returns simulated stock prices with Heston variance
#example input parameters:
#S0 =100, Vt0 = 0.012, r = 0.0272, tht = 0.012
#kpp = 1.98937, epp = 0.15, rho_sv = -0.9, t = 1, dt = 0.01
    # import S0           "Initial stock price"
    # import Vt0          "Initial variance"
    # import r            "Risk free interest rate"
    # import tht          "Long term variance"
    # import kpp          "Mean reversion rate for Heston variance"
    # import epp          "Volatility of Heston variance"
    # import rho_sv       "Instantaneous correlation of stock price and stock price variance"
    # import t            "Simulation time horizon"
    # import dt           "Discretization time step"
    # export S            "Heston stock prices, Drift Martingale Corrected"
    # export Vt           "Heston variance process"

from __future__ import division

import numpy as np
import numexpr as ne
from scipy.stats.distributions import norm

random_batch_size = 5

def paths(S0, interest_rate, V0, t, mean_reversion, mean_variance, vol_of_var, correlation, dt, npaths, callback=lambda x,y: None):
    neval = ne.evaluate
    norminv = norm.ppf
    exp  = np.exp
    log  = np.log
    sqrt = np.sqrt

    rho_sv = correlation
    r = interest_rate
    kpp = mean_reversion
    tht = mean_variance
    epp = vol_of_var

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

    for i in range(nrOfSteps):
        if not i % notify:
            print int(100*i / nrOfSteps),
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


def payoff(s, k):
    return np.maximum(s - k, 0)

def discount(x, r, t):
    return np.exp(-r*t)*x

def compose(l):
    def newf(*args, **kwargs):
        x = l.pop(*args, **kwargs)
        for f in l[:-1]:
            x = f(x)
        return x
    return newf

def plot(s, v):
    fig = pylab.figure()
    saxes = fig.add_subplot(211)
    vaxes = fig.add_subplot(212)
    saxes.plot(s)
    vaxes.plot(v)
    pylab.show()

def knockout120(s, state):
    state &= (s < 120)

def main():
    import pylab
    s = 100.0
    k = 99.0
    r = 0.06
    v = 0.04
    t = 1.0
    kappa = 1
    theta = v
    sigma = 0.0001
    rho = 0.0001
    dt = 0.001
    npaths = 1000
    s = paths(s, r, v, t, kappa, theta, sigma, rho, dt, npaths, callback=knockout120)
    # s = paths(s, r, v, t, kappa, theta, sigma, rho, dt, npaths)
    p = discount(payoff(s,k), r, t)
    print "Expected:", np.mean(p)
    print "Error:", 1.96*np.std(p) / np.sqrt(npaths)






if __name__ == '__main__':
    main()




zip(""" 4.81984886677 3.32390996286 1.58433557015 1.00544386651 0.540681642949 """.split(),
    [46.875, 23.4375, 11.71875, 5.859375, 2.9296875])


