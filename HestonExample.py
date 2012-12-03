#!/usr/bin/env python
"""Demonstration of 2D Heston using BTCS CTCS and Smoothed CTCS."""

from bisect import bisect_left
import sys

import numpy as np
from pylab import plot, title, xlabel, ylabel, legend, show, ion, ioff
import scipy.linalg as spl
import scipy.sparse as sps

from FiniteDifference import FiniteDifferenceEngine as FDE
from FiniteDifference import utils
from FiniteDifference.Grid import Grid
from FiniteDifference.Option import BlackScholesOption
from FiniteDifference.heston import HestonFiniteDifferenceEngine
from FiniteDifference.heston import HestonOption, hs_call_vector
from FiniteDifference.visualize import surface


# Debug imports
# from FiniteDifference.visualize import fp, anim, wireframe


H = HestonOption( spot=100
                 , strike=100
                 , interest_rate=0.03
                 , volatility = 0.2
                 , tenor=1.0
                 , mean_reversion = 1
                 , mean_variance = 0.04
                 , vol_of_variance = 0.4
                 , correlation = 0.6
                 )

dt=1/100.0
spot_max = 1500.0
var_max = 10.0

nspots = 100
nvols = 100

spotdensity = 7.0  # infinity is linear?
varexp = 4

spots = utils.sinh_space(H.strike, spot_max, spotdensity, nspots)
vars = utils.exponential_space(0.00, H.variance.value, var_max, varexp, nvols)
# vars = [v0]
# spots = np.linspace(0.0, spot_max, nspots)
# vars = np.linspace(0.0, var_max, nvols)
# plot(spots); title("Spots"); show()
# plot(vars); title("Vars"); show()

# H.strike = spots[min(abs(spots - H.strike)) == abs(spots - H.strike)][0]
# H.spot = spots[min(abs(spots - H.spot)) == abs(spots - H.spot)][0]
def init(spots, vars):
    return np.maximum(0, spots - H.strike)
Gi = Grid(mesh=(spots, vars), initializer=init)
G = Gi.copy()
V_init = G.domain[-1].copy()

trims = (H.strike * .2 < spots) & (spots < H.strike * 2.0)
trimv = (0.0 < vars) & (vars < 1)  # v0*2.0)
# trims = slice(None)
# trimv = slice(None)

# Does better without upwinding here
up_or_down_spot = 'up'
up_or_down_var = 'down'
flip_idx_var = bisect_left(vars, H.mean_variance)
flip_idx_spot = 1

tr = lambda x: x[trims, :][:, trimv]
tr3 = lambda x: x[:, trims, :][:, :, trimv]

ids = np.isclose(spots[trims], H.spot)
idv = np.isclose(vars[trimv], H.variance.value)
dss = np.hstack((np.nan, np.diff(spots)))
dvs = np.hstack((np.nan, np.diff(vars)))

BADANALYTICAL = False

bs = BlackScholesOption(spot=spots[:, np.newaxis],
                        strike=H.strike,
                        interest_rate=H.interest_rate.value,
                        variance=vars[np.newaxis, :],
                        tenor=H.tenor).analytical

utils.tic("Heston Analytical:\t")
hs = hs_call_vector(spots, H.strike, H.interest_rate.value, np.sqrt(vars),
             H.tenor, H.mean_reversion, H.mean_variance, H.vol_of_variance, H.correlation)
utils.toc()
hs = np.nan_to_num(hs)
if max(hs.flat) > spots[-1] * 2:
    BADANALYTICAL = True
    print "Warning: Analytical solution looks like trash."

if len(sys.argv) > 1:
    if sys.argv[1] == '0':
        print "Bail out with arg 0."
        sys.exit()

def mu_s(t, *dim):
    return H.interest_rate.value * dim[0]
def gamma2_s(t, *dim):
    return 0.5 * dim[1] * dim[0]**2
def mu_v(t, *dim):
    return H.mean_reversion * (H.mean_variance - dim[1])
def gamma2_v(t, *dim):
    return 0.5 * H.vol_of_variance**2 * dim[1]
def cross(t, *dim):
    return H.correlation * H.vol_of_variance * dim[0] * dim[1]

coeffs = {()   : lambda t: -H.interest_rate.value,
          (0,) : mu_s,
          (0,0): gamma2_s,
          (1,) : mu_v,
          (1,1): gamma2_v,
          (0,1): cross
          }

bounds = {      # D: U = 0              VN: dU/dS = 1
        (0,)  : ((0, lambda *args: 0.0), (1, lambda *args: 1.0)),
                # D: U = 0              Free boundary
        (0,0) : ((0, lambda *args: 0.0), (None, lambda *x: None)),
                # Free boundary at low variance
        (1,)  : ((None, lambda *x: None),
                # D intrinsic value at high variance
                (0.0, lambda t, *dim: np.maximum(0.0, dim[0]-H.strike))),
                # Free boundary
        (1,1) : ((None, lambda *x: None),
                # D intrinsic value at high variance
                (0.0, lambda t, *dim: np.maximum(0.0, dim[0]-H.strike)))}

# schemes = {(1,) : ({"scheme": "center"}, {"scheme": "backward", "from" : flip_idx_var})}


utils.tic("Building FDE Engine")
# F = FDE.FiniteDifferenceEngineADI(G, coefficients=coeffs, boundaries=bounds, schemes=schemes, force_bandwidth=(-2, 2))
F = HestonFiniteDifferenceEngine(H, nspots=nspots, nvols=nvols, flip_idx_spot=flip_idx_spot, flip_idx_var=flip_idx_var)
utils.toc()

L1_ = []
R1_ = []
utils.tic("Building As(s):\t")
print "(Up/Down)wind from:", flip_idx_spot
As_ = utils.nonuniform_complete_coefficients(dss, up_or_down=up_or_down_spot,
                                             flip_idx=flip_idx_spot)[0]
Ass_ = utils.nonuniform_complete_coefficients(dss)[1]
# As_, Ass_ = utils.nonuniform_forward_coefficients(dss)
assert(not np.isnan(As_.data).any())
assert(not np.isnan(Ass_.data).any())
mu_sfunc = mu_s
gamma2_sfunc = gamma2_s
for j, v in enumerate(vars):
    # Be careful not to overwrite our operators
    As, Ass = As_.copy(), Ass_.copy()
    m = 2

    mu_s = H.interest_rate.value * spots
    # mu_s = mu_sfunc(0, spots, v)
    # gamma2_s = gamma2_sfunc(0, spots, v)
    gamma2_s = 0.5 * v * spots ** 2
    for i, z in enumerate(mu_s):
        # print z, coeffs[0,](0, spots[i])
        assert z == coeffs[0,](0, spots[i])
    for i, z in enumerate(gamma2_s):
        # print z, coeffs[0,0](0, spots[i], v)
        assert z == coeffs[0,0](0, spots[i], v)

    Rs = np.zeros(nspots)
    Rs[-1] = 1

    As.data[m - 2, 2:] *= mu_s[:-2]
    As.data[m - 1, 1:] *= mu_s[:-1]
    As.data[m, :] *= mu_s
    As.data[m + 1, :-1] *= mu_s[1:]
    As.data[m + 2, :-2] *= mu_s[2:]

    Rs *= mu_s

    Rss = np.zeros(nspots)
    Rss[-1] = 2 * dss[-1] / dss[-1] ** 2

    Ass.data[m, -1] = -2 / dss[-1] ** 2
    Ass.data[m + 1, -2] = 2 / dss[-1] ** 2

    Ass.data[m - 2, 2:] *= gamma2_s[:-2]
    Ass.data[m - 1, 1:] *= gamma2_s[:-1]
    Ass.data[m, :] *= gamma2_s
    Ass.data[m + 1, :-1] *= gamma2_s[1:]
    Ass.data[m + 2, :-2] *= gamma2_s[2:]

    Rss *= gamma2_s

    L1_.append(As.copy())
    L1_[j].data += Ass.data
    L1_[j].data[m, :] -= 0.5 * H.interest_rate.value
    L1_[j].data[m, 0] = -1

    R1_.append((Rs + Rss).copy())
    R1_[j][0] = 0
utils.toc()

mu_v = H.mean_reversion * (H.mean_variance - vars)
gamma2_v = 0.5 * H.vol_of_variance ** 2 * vars

L2_ = []
R2_ = []
utils.tic("Building Av(v):\t")
print "(Up/Down)wind from:", flip_idx_var
# Avc_, Avvc_ = utils.nonuniform_center_coefficients(dvs)
Av_ = utils.nonuniform_complete_coefficients(dvs, up_or_down=up_or_down_var,
                                             flip_idx=flip_idx_var)[0]
Avv_ = utils.nonuniform_complete_coefficients(dvs)[1]
assert(not np.isnan(Av_.data).any())
assert(not np.isnan(Avv_.data).any())
for i, s in enumerate(spots):
    # Be careful not to overwrite our operators
    Av, Avv = Av_.copy(), Avv_.copy()

    m = 2

    Av.data[m - 2, 2] = -dvs[1] / (dvs[2] * (dvs[1] + dvs[2]))
    Av.data[m - 1, 1] = (dvs[1] + dvs[2]) / (dvs[1] * dvs[2])
    Av.data[m, 0] = (-2 * dvs[1] - dvs[2]) / (dvs[1] * (dvs[1] + dvs[2]))

    Av.data[m - 2, 2:] *= mu_v[:-2]
    Av.data[m - 1, 1:] *= mu_v[:-1]
    Av.data[m, :] *= mu_v
    Av.data[m + 1, :-1] *= mu_v[1:]
    Av.data[m + 2, :-2] *= mu_v[2:]

    Rv = np.zeros(nvols)
    Rv *= mu_v

    Avv.data[m - 1, 1] = 2 / dvs[1] ** 2
    Avv.data[m, 0] = -2 / dvs[1] ** 2

    Avv.data[m - 2, 2:] *= gamma2_v[:-2]
    Avv.data[m - 1, 1:] *= gamma2_v[:-1]
    Avv.data[m, :] *= gamma2_v
    Avv.data[m + 1, :-1] *= gamma2_v[1:]
    Avv.data[m + 2, :-2] *= gamma2_v[2:]

    Rvv = np.zeros(nvols)
    Rvv[0] = 2 * dvs[1] / dvs[1] ** 2
    Rvv *= gamma2_v

    L2_.append(Av.copy())
    L2_[i].data += Avv.data
    L2_[i].data[m, :] -= 0.5 * H.interest_rate.value

    L2_[i].data[m, -1] = -1  # This is to cancel out the previous value so we can
                        # set the dirichlet boundary condition using R.
                        # Then we have U_i + -U_i + R

    R2_.append(Rv + Rvv)
    R2_[i][-1] = np.maximum(0, s - H.strike)

utils.toc(':  \t')


def force_boundary(V, values=None, t=None):
    # m1 = hs_call(spots, H.strike, H.interest_rate, sqrt(np.array((vars[0], vars[-1]))), t, H.mean_reversion, H.mean_variance, H.vol_of_variance, rho)
    # m2 = hs_call(np.array((spots[0], spots[-1])), H.strike, H.interest_rate, sqrt(vars), t, H.mean_reversion, H.mean_variance, H.vol_of_variance, rho)
    m = values
    m1 = m2 = m
    V[0, :] = m2[0, :]  # top
    V[:, 0] = m1[:, 0]  # left
    V[-1, :] = m2[-1, :]  # bottom
    V[:, -1] = m1[:, -1]  # right


def impl(V, L1, R1x, L2, R2x, dt, n, initial=None, callback=None):
    V = V.copy()

    # L1i = flatten_tensor(L1)
    L1i = L1.copy()
    R1 = np.array(R1x).T

    # L2i = flatten_tensor(L2)
    L2i = L2.copy()
    R2 = np.array(R2x)

    m = 2

    # L  = (As + Ass - H.interest_rate*np.eye(nspots))*-dt + np.eye(nspots)
    L1i.data *= -dt
    L1i.data[m, :] += 1
    R1 *= -dt

    L2i.data *= -dt
    L2i.data[m, :] += 1
    R2 *= -dt

    offsets1 = (abs(min(L1i.offsets)), abs(max(L1i.offsets)))
    offsets2 = (abs(min(L2i.offsets)), abs(max(L2i.offsets)))

    dx = np.gradient(spots)[:,np.newaxis]
    dy = np.gradient(vars)
    Y, X = np.meshgrid(vars, spots)
    gradgrid = dt * coeffs[(0,1)](0, X, Y) / (dx * dy)
    gradgrid[:,0] = 0; gradgrid[:,-1] = 0
    gradgrid[0,:] = 0; gradgrid[-1,:] = 0

    print_step = max(1, int(n / 10))
    to_percent = 100.0 / n
    utils.tic("Impl:\t")
    for k in xrange(n):
        if not k % print_step:
            if np.isnan(V).any():
                print "Impl fail @ t = %f (%i steps)" % (dt * k, k)
                return crumbs
            print int(k * to_percent),
        if callback is not None:
            callback(V, ((n - k) * dt))
        Vsv = np.gradient(np.gradient(V)[0])[1] * gradgrid
        # Vsv = 0.0
        V = spl.solve_banded(offsets2, L2i.data,
                             (V + Vsv - R2).flat, overwrite_b=True).reshape(V.shape)
        V = spl.solve_banded(offsets1, L1i.data,
                             (V - R1).T.flat, overwrite_b=True).reshape(V.shape[::-1]).T
    crumbs.append(V.copy())
    utils.toc(':  \t')
    return crumbs


def flatten_tensor(mats):
    diags = np.hstack([x.data for x in mats])
    flatmat = sps.dia_matrix((diags, mats[0].offsets), shape=(diags.shape[1], diags.shape[1]))
    return flatmat


def douglas(V, L1, R1x, L2, R2x, dt, n, initial=None, callback=None):
    V = V.copy()
    theta = 0.5
    # dt *= 0.5

    # L1e = flatten_tensor(L1)
    L1e = L1.copy()
    L1i = L1e.copy()
    R1 = np.array(R1x).T

    # L2e = flatten_tensor(L2)
    L2e = L2.copy()
    L2i = L2e.copy()
    R2 = np.array(R2x)

    # print "L var"
    # fp(L2e.data, 2)
    # print "FDE op var"
    # fp(F.operators[1].data, 2)

    # print "diff"
    # fp(F.operators[1].data - L2e.data, 2, 'f')

    # assert np.allclose(F.operators[1].data, L2e.data)

    m = 2

    # L  = (As + Ass - H.interest_rate*np.eye(nspots))*-dt + np.eye(nspots)

    L1i.data *= -theta*dt
    L1i.data[m, :] += 1
    # R1 *= dt

    L2i.data *= -theta*dt
    L2i.data[m, :] += 1
    # R2 *= dt

    offsets1 = (abs(min(L1i.offsets)), abs(max(L1i.offsets)))
    offsets2 = (abs(min(L2i.offsets)), abs(max(L2i.offsets)))

    dx = np.gradient(spots)[:,np.newaxis]
    dy = np.gradient(vars)
    X, Y = [dim.T for dim in np.meshgrid(spots, vars)]
    gradgrid = coeffs[(0,1)](0, X, Y) / (dx*dy)
    gradgrid[:,0] = 0; gradgrid[:,-1] = 0
    gradgrid[0,:] = 0; gradgrid[-1,:] = 0

    print_step = max(1, int(n / 10))
    to_percent = 100.0 / n
    utils.tic("Douglas:\t")
    R = R1 + R2
    normal_shape = V.shape
    transposed_shape = V.T.shape
    for k in xrange(n):
        if not k % print_step:
            if np.isnan(V).any():
                print "Douglas fail @ t = %f (%i steps)" % (dt * k, k)
                return crumbs
            print int(k * to_percent),
        if callback is not None:
            callback(V, ((n - k) * dt))

        Vsv = np.gradient(np.gradient(V)[0])[1] * gradgrid

        # V12 = (V
                # + Vsv
                # + (1-theta)*dt*L1e.dot(V.T.flat).reshape(transposed_shape).T
                # + (1-theta)*dt*L2e.dot(V.flat).reshape(normal_shape)
                # + dt * R)

        # V1 = spl.solve_banded(offsets2, L2i.data, V12.flat, overwrite_b=True).reshape(normal_shape)
        # V  = spl.solve_banded(offsets1, L1i.data, V1.T.flat, overwrite_b=True).reshape(transposed_shape).T

        V1 = (L1e.dot(V.T.flat).reshape(transposed_shape)).T
        V2 = (L2e.dot(V.flat).reshape(normal_shape))
        Y0 = V + dt*(Vsv + V1 + V2 + R)

        V1 = Y0 - theta * dt * L1e.dot(V.T.flat).reshape(transposed_shape).T
        Y1 = spl.solve_banded(offsets1, L1i.data, V1.T.flat, overwrite_b=True).reshape(transposed_shape).T

        V2 = Y1 - theta * dt * L2e.dot(V.flat).reshape(normal_shape)
        Y2 = spl.solve_banded(offsets2, L2i.data, V2.flat, overwrite_b=True).reshape(normal_shape)
        V = Y2


        crumbs.append(V.copy())
    utils.toc(':  \t')
    return crumbs


# Trim for plotting
front = 1
back = 1
line_width = 2
          # exp  imp   cr   smo
markers = ['--', '--', ':', '--']


def p_1D(V, analytical, spots, vars, marker_idx, label):
    if BADANALYTICAL:
        label += " - bad analytical!"
    else:
        label += " - ||V - V*|| = %.2e" % np.linalg.norm(V-analytical)
    plot((spots / H.strike * 100)[front:-back],
         (V - analytical)[front:-back],
         markers[marker_idx], lw=line_width, label=label)
    title("Error in Price")
    xlabel("% of strike")
    ylabel("Error")
    legend(loc=0)


def p_absolute_error(V, analytical, spots, vars, marker_idx=0, label=""):
    surface(V - analytical, spots, vars)
    if BADANALYTICAL:
        label += " - bad analytical!"
    else:
        label += " - ||V - V*|| = %.2e" % np.linalg.norm(V-analytical)
    title("Error in Price (%s)" % label)
    xlabel("Var")
    ylabel("% of strike")
    show()


def p_relative_error(V, analytical, spots, vars, marker_idx=0, label=""):
    surface((V - analytical) / analytical, spots, vars)
    if BADANALYTICAL:
        label += " - bad analytical!"
    else:
        label += " - ||V - V*|| = %.2e" % np.linalg.norm(V-analytical)
    title("Relative Error in Price (%s)" % label)
    xlabel("Var")
    ylabel("% of strike")
    show()


p = p_absolute_error; V = None
evis = lambda V=V, p=p_absolute_error: p(V, hs, spots, vars, 0, "")
evis2 = lambda V=V, p=p_absolute_error: p(tr(V), tr(hs), spots[trims], vars[trimv], 0, "")
vis = lambda V=V, p=p_absolute_error: p(V, 0, spots, vars, 0, "")
vis2 = lambda V=V, p=p_absolute_error: p(tr(V), 0, spots[trims], vars[trimv], 0, "")

L1 = F.operators[0].D
L2 = F.operators[1].D
R1 = F.operators[0].R.reshape(V_init.T.shape)
R2 = F.operators[1].R.reshape(V_init.shape)
# print "Old"
# fp(np.array(R1_))
# print "new"
# fp(R1)
# print "diff"
# fp(R1 - np.array(R1_), fmt='e')
assert (flatten_tensor(L1_).data == L1.data).all()
assert (flatten_tensor(L2_).data == L2.data).all()
assert np.array(R1).shape == np.array(R1_).shape
assert np.array(R2).shape == np.array(R2_).shape
assert np.allclose(np.array(R1), np.array(R1_))
assert np.allclose(np.array(R2), np.array(R2_))
assert (V_init == F.grid.domain).all()

# print "OK"

# sys.exit()
# Vs = impl(V_init, L1, R1, L2, R2,
          # dt, int(H.tenor / dt), crumbs=[V_init]
          # # , callback=lambda v, H.tenor: force_boundary(v, hs)
          # )
# Vi = Vs[-1]
# print tr(Vi)[ids, idv] - tr(hs)[ids, idv]
# Vs = F.solve_implicit(H.tenor/dt, dt, initial=None, callback=None)
# Vfi = Vs[-1]
# print tr(Vfi)[ids, idv] - tr(hs)[ids, idv]

# Vs = douglas(V_init, L1, R1, L2, R2,
           # dt, int(H.tenor / dt), crumbs=[V_init]
           # # , callback=lambda v, H.tenor: force_boundary(v, hs)
           # )
# Vd = Vs[-1]
# print tr(Vd)[ids, idv] - tr(hs)[ids, idv]
# Vs = F.solve_douglas(H.tenor/dt, dt, initial=None, callback=None)
Vfd = F.smooth(H.tenor/dt, dt, initial=None, callback=None, scheme=F.solve_douglas)
print tr(Vfd)[ids, idv] - tr(hs)[ids, idv]
print
print tr(Vfd)[ids, idv], tr(hs)[ids, idv]
print F.price, F.option.analytical
print "Price difference:", F.price - tr(F.grid.domain[-1])[ids,idv]
print
F.grid.reset()

# Vs = F.solve_craigsneyd(H.tenor/dt, dt, initial=None, callback=None)
# Vfcs = Vs[-1]
# print tr(Vfcs)[ids, idv] - tr(hs)[ids, idv]

# Vs = F.solve_craigsneyd2(H.tenor/dt, dt, initial=None, callback=None)
# Vfcs2 = Vs[-1]
# print tr(Vfcs2)[ids, idv] - tr(hs)[ids, idv]

# Vs = F.solve_hundsdorferverwer(H.tenor/dt, dt, initial=None, callback=None)
Vfhv = F.smooth(H.tenor/dt, dt, initial=None, scheme=F.solve_hundsdorferverwer, callback=None)
F.grid.reset()
print tr(Vfhv)[ids, idv] - tr(hs)[ids, idv]

# Vs = F.solve_john_adi(H.tenor/dt, dt, initial=None, callback=None)
# Vfc = Vs[-1]
# print tr(Vfc)[ids, idv] - tr(hs)[ids, idv]


# Rannacher smoothing to damp oscilations at the discontinuity
# smoothing_steps = 2
# Vs = impl(V_init, L1, R1, L2, R2,
          # dt/2, 2*smoothing_steps, crumbs=[V_init]
          # # , callback=lambda v, H.tenor: force_boundary(v, hs)
          # )
# Vs = douglas(Vs[-1], L1, R1, L2, R2,
          # dt, int(H.tenor / dt) - smoothing_steps, crumbs=Vs
          # # , callback=lambda v, H.tenor: force_boundary(v, hs)
          # )
# Vr = Vs[-1]
# print tr(Vr)[ids, idv] - tr(hs)[ids, idv]
# Vs = F.smooth(H.tenor/dt, dt, initial=None, callback=None,
              # smoothing_steps=smoothing_steps)
# Vfr = Vs[-1]
# print tr(Vfr)[ids, idv] - tr(hs)[ids, idv]

ion()
# p(tr(Vi), tr(hs), spots[trims], vars[trimv], 1, "impl")
# p(tr(Vfi), tr(hs), spots[trims], vars[trimv], 1, "FDE impl")
# p(tr(Vc), tr(hs), spots[trims], vars[trimv], 2, "crank")
# p(tr(Vfc), tr(hs), spots[trims], vars[trimv], 2, "FDE crank")
# p(tr(Vr), tr(hs), spots[trims], vars[trimv], 3, "smooth")
# p(tr(Vfr), tr(hs), spots[trims], vars[trimv], 3, "FDE smooth")
# p(tr(Vd), tr(hs), spots[trims], vars[trimv], 3, "douglas")
# p(tr(Vfd), tr(hs), spots[trims], vars[trimv], 3, "FDE douglas")
# p(tr(Vfcs), tr(hs), spots[trims], vars[trimv], 3, "FDE craigsneyd")
# p(tr(Vfcs2), tr(hs), spots[trims], vars[trimv], 3, "FDE craigsneyd2")
# p(tr(Vfhv), tr(hs), spots[trims], vars[trimv], 3, "FDE hundsdorferverwer")
ioff()
show()
# p(V_init, hs, spots, vars, 1, "impl")
# p(Vc, hs, spots, vars, 1, "crank")
# p(Vr, hs, spots, vars, 1, "smooth")


if p is p_1D:
    show()
