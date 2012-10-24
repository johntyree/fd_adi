
from pylab import *
import scipy.sparse as sps
import scipy.linalg as spl

def init(xs, vs, k):
    u = np.ones((len(xs),len(vs))).T * xs
    u = u.T
    return np.maximum(0, u-k)

def exponential_space(low, exact, high, ex, n):
    v = np.zeros(n);
    l = pow(low,1./ex);
    h = pow(high,1./ex);
    x = pow(exact,1./ex);
    dv = (h - l) / (n-1);
    j = 0

    d = 1e100
    for i in range(n):
        if (i*dv > x):
        # if abs(i*dv - x) < d:
            # d = abs(i*dv - x)
            j = i-1
            break
    if (j == 0):
        print "Did not find thingy."
        assert(j != 0)
    dx = x - j*dv;
    h += (n-1) * dx/j;
    dv = (h - l) / (n-1);
    for i in range(n):
        v[i] = pow(i*dv, ex)
    return v;

def cubic_sigmoid_space(exact, high, density, n):
    if density == 0:
        # return hstack((linspace(exact - (exact - high), exact, n//2)[:-1]
                      # ,linspace(exact, high, n//2)))
        return linspace(exact - (high - exact), high, n)
    y = zeros(n)

    dx = 1.0/(n-1)
    scale = (float(high)-exact)/(density**3 + density)
    for i in range(n):
        x = (2*(i*dx)-1)*density
        y[i] = exact + (x**3+x)*scale

    return y

def filterprint(A, prec=1, fmt="f", predicate=lambda x: x == 0, blank='- '):
    '''
    Pretty print a NumPy array, hiding values which match a predicate
    (default: x == 0). predicate must be callable.
    '''
    if A.ndim == 1:
        A = A[:,newaxis]
    tmp = "% .{0}{1}".format(prec, fmt)
    xdim, ydim = np.shape(A)
    pad = max(len(tmp % x) for x in A.flat)
    fmt = "% {pad}.{prec}{fmt}".format(pad=pad, prec=prec, fmt=fmt)
    bstr = "{:>{pad}}".format(blank, pad=pad)
    for i in range(xdim):
        for j in range(ydim):
            if not predicate(A[i,j]):
                print fmt % A[i,j],
            else:
                print bstr,
        print
    return


spot = 100
maxspot = 120
spotdensity = 0
strike = 99.
kappa = 1.
lam = 0.
r = 0.06
sigma = 0
tau = 1
var = 0.2**2
maxvar = 0.2
varexp = 1
theta = var
rho = 0.
dt = 1/10000.
sub = 0
mid = 1
sup = 2

nspots = 500
nvols = 0

spots = cubic_sigmoid_space(spot, maxspot, spotdensity, nspots+1)
dss = np.hstack((nan, np.diff(spots)))

# vars = exponential_space(0, var, maxvar, varexp, nvols+1)
vars = array([var])
dvs = np.hstack((nan, np.diff(vars)))


ids = min(abs(spots - spot)) == abs(spots - spot)
idv = min(abs(vars - var)) == abs(vars - var)


Ui = init(spots,vars,strike)
U  = init(spots,vars,strike)

def cdiff(a, s):
    c = empty_like(a)
    c[:-1]   = diff(a) / diff(s)
    c[1:]   += (diff(a[::-1]) / diff(s[::-1]))[::-1]
    c[1:-1] /= 2
    return c

# A = zeros(((nspots+1)*(nvols+1), (nspots+1)*(nvols+1)))

def nonuniform_coefficients(deltas):
    d = deltas
    a = empty((2,len(d)))
    a[:,0] = nan
    a[:,-1] = nan
    b = a.copy()
    c = a.copy()
    for j in range(1,len(d)-1):
        a[0][j] = -d[j+1] / (d[j]*(d[j]+d[j+1]));
        b[0][j] = (-d[j] + d[j+1]) / (d[j]*d[j+1]);
        c[0][j] = d[j] / (d[j+1]*(d[j]+d[j+1]));

        a[1][j] = 2 / (d[j]*(d[j]+d[j+1]));
        b[1][j] = -2 / (d[j]*d[j+1]);
        c[1][j] = 2 / (d[j]*(d[j]+d[j+1]));
    return a,b,c



a1,b1,c1 = nonuniform_coefficients(dss)
d_ds   = np.zeros((nspots+1, nspots+1))
d2_dss = np.zeros((nspots+1, nspots+1))
Rs  = np.zeros((nspots+1))
Rss = np.zeros((nspots+1))
I = np.eye(nspots+1)

# Treating S dimension
d_ds  [0,:] = 0
d2_dss[0,:] = 0
for i in range(1,nspots): # interior points
   d_ds[i,i-1] = a1[0][i]
   d_ds[i,i+0] = b1[0][i]
   d_ds[i,i+1] = c1[0][i]

   d2_dss[i,i-1] = a1[1][i]
   d2_dss[i,i+0] = b1[1][i]
   d2_dss[i,i+1] = c1[1][i]
d_ds  [nspots,:] = 0
d2_dss[nspots,:] = 0
d2_dss[nspots,i-1] =  2/dss[-1]**2
d2_dss[nspots,i+0] = -2/dss[-1]**2
Rs[:] = 0
Rs[-1] = 1
Rss[:] = 0
Rss[-1] = 2*dss[-1] / dss[2]**2
d2_dss[nspots,:] = 0

v = var
mu_s = r * spots
gamma2_s = 0.5 * v * spots**2
A1 = mu_s[:,newaxis] * d_ds + gamma2_s[:,newaxis] * d2_dss - r*I
R1 = (mu_s * Rs   + gamma2_s * Rss)[:,newaxis]

# a2,b2,c2 = nonuniform_coefficients(dvs)
# d_dv  = np.zeros((nvols+1, nvols+1))
# d2_dvv = np.zeros((nvols+1, nvols+1))
# Rv = np.zeros(nvols+1)
# Rvv = np.zeros(nvols+1)
# I = np.eye(nspots+1)

# Treating V dimension
# d_dv [0,:] = 0
# d2_dvv[0,:] = 0
# d_dv[0,0] = -1/dvs[1]
# d_dv[0,1] =  1/dvs[1]
# for j in range(1,nvols): # interior points
   # d_dv[j,j-1] = a2[0][j]
   # d_dv[j,j+0] = b2[0][j]
   # d_dv[j,j+1] = c2[0][j]

   # d2_dvv[j,j-1] = a2[1][j]
   # d2_dvv[j,j+0] = b2[1][j]
   # d2_dvv[j,j+1] = c2[1][j]
# Rv[:] = 0
# Rvv[:] = 0
# d_dv [nvols,:] = 0
# d2_dvv[nvols,:] = 0


def step_(U, dt):
    U = (I + dt*A1).dot(U) + dt*R1
    # U = solve(I - dt*A1, U) - dt*R1
    return U

def step(U, t):
    for i in range(int(t/dt)):
        if not i % int(t/dt):
            print "%i%%" % (int(i* dt/t * 100))
        U = step_(U, dt)
        if (np.isnan(U).any()):
            print "Nan in domain after %i steps. Time: %f" % (i, i * dt/t)
            break
    return U

res = step(U,1)
print res[ids]

sys.exit()



# Axt  = sps.dia_matrix((Ax,
                     # ((nvols+1), 0, -(nvols+1))),
                    # shape=((nspots+1)*(nvols+1), (nspots+1)*(nvols+1)))
# Axxt = sps.dia_matrix((Axx,
                     # ((nvols+1), 0, -(nvols+1))),
                    # shape=((nspots+1)*(nvols+1), (nspots+1)*(nvols+1)))
# A1 = sps.dia_matrix((I + 0.5*dt * (Ax+Axx-0.5*r*I),
                     # ((nvols+1), 0, -(nvols+1))),
                    # shape=((nspots+1)*(nvols+1), (nspots+1)*(nvols+1)))




# # A2 = Av + Avv
# A2 = sps.dia_matrix((I + 0.5*dt * (Av+Avv-0.5*r*I),
                    # (1, 0, -1)),
                    # shape=((nspots+1)*(nvols+1), (nspots+1)*(nvols+1)))

# full = np.vstack([Ax[0]+Axx[0],
                 # Av[0]+Avv[0],
                 # Ax[1]+Axx[1]+Av[1]+Avv[1],
                 # Av[2]+Avv[2],
                 # Ax[2]+Axx[2]])


# I   = np.zeros((5, (nspots+1)*(nvols+1)))
# I[2,:] = 1
# A0 = sps.dia_matrix((I + dt * (full-r*I),
                    # ((nvols+1), 1, 0, -1, -(nvols+1))),
                    # shape=((nspots+1)*(nvols+1), (nspots+1)*(nvols+1)))



def D(dim, boundary=False):
    domain = np.arange(dim)
    return discrete_first(domain, boundary)

def D2(dim, boundary=False):
    domain = np.arange(dim)
    return discrete_second(domain, boundary)

def discrete_first(domain, boundary=False):
    '''Discrete first derivative operator with no boundary.'''
    operator = np.zeros((len(domain), len(domain)))
    (xdim, ydim) = np.shape(operator)
    if boundary:
        operator[0,1] = 0.5
    else:
        xstart, xend = 1, xdim-1
        ystart, yend = 1, ydim-1
    for i in xrange(xstart, xend):
        for j in xrange(1, ydim-1):
            if i == j:
                operator[i][j-1] = -0.5
                operator[i][j+1] =  0.5
    # operator[-1][-2] = -1
    # operator[-1][-1] =  1
    return operator

def discrete_second(domain, boundary=False):
    '''Discrete second derivative operator with no boundary.'''
    operator = np.zeros((len(domain), len(domain)))
    (xdim, ydim) = np.shape(operator)
    if boundary:
        xstart, xend = 0, xdim
        ystart, yend = 0, ydim
    else:
        xstart, xend = 1, xdim-1
        ystart, yend = 1, ydim-1
    for i in xrange(xstart, xend):
        for j in xrange(ystart, yend):
            if i == j:
                operator[i][j-1] =  1
                operator[i][j  ] = -2
                operator[i][j+1] =  1
    return operator






# print("Ax")
# filterprint(Ax)
# print("Axx")
# filterprint(Axx)
# print("Av")
# filterprint(Av)
# print("Avv")
# filterprint(Avv, isnan)


# A1_exp = sps.dia_matrix(np.identity((nspots+1)*(nvols+1)) + dt * A1)
# A2_exp = sps.dia_matrix(np.identity((nspots+1)*(nvols+1)) + dt * A2)

