# Author: Bhuvi
# MCRB tools: the misspecified Cramer-Rao bound (sandwich) for robust losses in linear regression
# Used by sim_J_mcrb_regression.py, sim_K_sandwich_coverage.py and tests/test_mcrb.py
#
# Model: y_i = x_i^T theta + w_i, w_i iid from the TRUE noise p, fixed design X (N x d).
# A loss rho(r) is the negative log-density of an ASSUMED noise, f(r) proportional to exp(-rho(r)).
# The estimator is theta_hat = argmin sum_i rho(y_i - x_i^T theta), i.e. the mismatched MLE (MML).
#
# SIGN CONVENTION (fixed here, used everywhere in the project code):
#   psi(r)  = rho'(r)                       (the derivative of the loss, an odd function for every loss here)
#   lam     = d/dt E_p[psi(w + t)] at t=0   (= E_p[psi'(w)] when psi is smooth, = 2 p(0) for MAE)
#   kappa   = E_p[psi(w)^2]
#   A = lam * X^T X  (positive definite),   B = kappa * X^T X
#   MCRB = A^-1 B A^-1 = (kappa / lam^2) (X^T X)^-1
# In Fortunati et al. 2017 notation, script-A = E_p[Hessian of ln f] = -A and script-B = B, so
# script-A^-1 script-B script-A^-1 = A^-1 B A^-1: the two minus signs cancel and the bound is the same.
# Under correct specification (rho = -ln p) we get kappa = lam = I_loc and MCRB = CRB = (X^T X)^-1 / I_loc.
#
# For symmetric noise and an odd psi, E_p[psi(w)] = 0, so the pseudo-true value of theta is the true theta
# for every loss (Regime A of the plan). The scale used inside a loss is set by the rule in loss_scale().

import numpy as np
from scipy import stats, integrate, optimize

# ----------------------------------------------------------------------------------------------------
# True noise families. Each has: sample, pdf, cdf, score u(w) = -d/dw ln p(w), Fisher info, variance
# ----------------------------------------------------------------------------------------------------

class Noise:
    name = "noise"
    breaks = (-50.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 50.0) # quadrature break points

    def expect(self, g):
        # E_p[g(w)] by quadrature, split at the break points so heavy tails and kinks are handled
        pts = (-np.inf,) + self.breaks + (np.inf,)
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            total += integrate.quad(lambda w: g(w) * self.pdf(w), a, b, limit=500)[0]
        return total

    def fisher(self):
        # location Fisher information of one sample, I_loc = E[u(w)^2]
        return self.expect(lambda w: self.score(w)**2)

    def quartile(self):
        # upper quartile of w; for symmetric noise this is the population MAD (median of |w|)
        return float(self.ppf(0.75))


class Gaussian(Noise):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.name = "Gaussian"
        self.d = stats.norm(0, sigma)
    def sample(self, rng, size): return rng.normal(0, self.sigma, size)
    def pdf(self, w): return self.d.pdf(w)
    def cdf(self, w): return self.d.cdf(w)
    def ppf(self, q): return self.d.ppf(q)
    def score(self, w): return w / self.sigma**2
    def fisher(self): return 1 / self.sigma**2
    def variance(self): return self.sigma**2


class Laplace(Noise):
    def __init__(self, b=1.0):
        self.b = b
        self.name = "Laplace"
        self.d = stats.laplace(0, b)
    def sample(self, rng, size): return rng.laplace(0, self.b, size)
    def pdf(self, w): return self.d.pdf(w)
    def cdf(self, w): return self.d.cdf(w)
    def ppf(self, q): return self.d.ppf(q)
    def score(self, w): return np.sign(w) / self.b
    def fisher(self): return 1 / self.b**2
    def variance(self): return 2 * self.b**2


class StudentT(Noise):
    def __init__(self, nu=3.0, s=1.0):
        self.nu, self.s = nu, s
        self.name = "Cauchy" if nu == 1 else "Student-t%g" % nu
        self.d = stats.t(nu, 0, s)
    def sample(self, rng, size): return self.s * rng.standard_t(self.nu, size)
    def pdf(self, w): return self.d.pdf(w)
    def cdf(self, w): return self.d.cdf(w)
    def ppf(self, q): return self.d.ppf(q)
    def score(self, w): return (self.nu + 1) * w / (self.nu * self.s**2 + w**2)
    def fisher(self): return (self.nu + 1) / ((self.nu + 3) * self.s**2) # derived in report Section 6.2
    def variance(self): return self.s**2 * self.nu / (self.nu - 2) if self.nu > 2 else np.inf


class Contaminated(Noise):
    # w ~ (1 - eps) N(0, s0^2) + eps N(0, s1^2), the outlier model of sim_H
    def __init__(self, eps=0.1, s0=1.0, s1=5.0):
        self.eps, self.s0, self.s1 = eps, s0, s1
        self.name = "Contaminated"
    def sample(self, rng, size):
        out = rng.random(size) < self.eps
        return np.where(out, rng.normal(0, self.s1, size), rng.normal(0, self.s0, size))
    def pdf(self, w):
        return (1 - self.eps) * stats.norm.pdf(w, 0, self.s0) + self.eps * stats.norm.pdf(w, 0, self.s1)
    def cdf(self, w):
        return (1 - self.eps) * stats.norm.cdf(w, 0, self.s0) + self.eps * stats.norm.cdf(w, 0, self.s1)
    def ppf(self, q): return optimize.brentq(lambda w: self.cdf(w) - q, -100, 100)
    def score(self, w):
        # u(w) = w (r0/s0^2 + r1/s1^2) with responsibilities computed in the log domain (no 0/0 in the tails)
        l0 = np.log(1 - self.eps) + stats.norm.logpdf(w, 0, self.s0)
        l1 = np.log(self.eps) + stats.norm.logpdf(w, 0, self.s1)
        lp = np.logaddexp(l0, l1)
        return w * (np.exp(l0 - lp) / self.s0**2 + np.exp(l1 - lp) / self.s1**2)
    def variance(self): return (1 - self.eps) * self.s0**2 + self.eps * self.s1**2


def default_noises():
    return [Gaussian(), Laplace(), StudentT(3), StudentT(1), Contaminated(0.1, 1.0, 5.0)]

# ----------------------------------------------------------------------------------------------------
# Losses. psi = rho', dpsi = psi'. Each robust loss has one scale c, set by loss_scale().
# ----------------------------------------------------------------------------------------------------

class Loss:
    name = "loss"
    q_unit = 1.0 # upper quartile of the loss's own noise family at unit scale (used by the scale rule)
    def lam(self, noise, c):
        # slope of the expected score, lam = E_p[psi'(w)]
        return noise.expect(lambda w: self.dpsi(w, c))
    def kappa(self, noise, c):
        return noise.expect(lambda w: self.psi(w, c)**2)


class MSE(Loss):
    name = "MSE"
    def rho(self, r, c=1.0): return 0.5 * r**2
    def psi(self, r, c=1.0): return r
    def dpsi(self, r, c=1.0): return np.ones_like(r)
    def weight(self, r, c=1.0): return np.ones_like(r)
    def lam(self, noise, c): return 1.0
    def kappa(self, noise, c): return noise.variance()


class MAE(Loss):
    name = "MAE"
    def rho(self, r, c=1.0): return np.abs(r)
    def psi(self, r, c=1.0): return np.sign(r)
    def weight(self, r, c=1.0): return 1 / np.maximum(np.abs(r), 1e-9)
    def lam(self, noise, c):
        # psi' does not exist at 0; Huber's (1967) form: d/dt E[sign(w + t)] = 2 p(0)
        return 2 * noise.pdf(0.0)
    def kappa(self, noise, c): return 1.0


class Huber(Loss):
    name = "Huber"
    def __init__(self, k=1.345):
        self.k = k                      # delta = k * sigma, k = 1.345 gives 95% efficiency at the Gaussian
        self.q_unit = stats.norm.ppf(0.75) / k  # so that delta = k * MAD / 0.6745
    def rho(self, r, c): return np.where(np.abs(r) <= c, 0.5 * r**2, c * np.abs(r) - 0.5 * c**2)
    def psi(self, r, c): return np.clip(r, -c, c)
    def dpsi(self, r, c): return (np.abs(r) <= c).astype(float)
    def weight(self, r, c): return np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
    def lam(self, noise, c): return noise.cdf(c) - noise.cdf(-c)


class TLoss(Loss):
    # negative log-likelihood of a Student-t(nu) with scale c, up to a constant; nu = 1 is the Cauchy loss
    def __init__(self, nu=3.0):
        self.nu = nu
        self.name = "Cauchy-loss" if nu == 1 else "t%g-loss" % nu
        self.q_unit = stats.t.ppf(0.75, nu)
    def rho(self, r, c): return 0.5 * (self.nu + 1) * np.log1p(r**2 / (self.nu * c**2))
    def psi(self, r, c): return (self.nu + 1) * r / (self.nu * c**2 + r**2)
    def dpsi(self, r, c): return (self.nu + 1) * (self.nu * c**2 - r**2) / (self.nu * c**2 + r**2)**2
    def weight(self, r, c): return (self.nu + 1) / (self.nu * c**2 + r**2) # EM / IRLS weight (Lange et al.)


def default_losses():
    return [MSE(), MAE(), Huber(1.345), TLoss(3), TLoss(1)]


def loss_scale(loss, mad):
    # Scale rule (fixed before any simulation, see the risk table of the plan):
    # each loss's scale is the value that makes it the exact MLE when its own noise family is true,
    # estimated from the MAD:  c = MAD / q_unit.
    #   Huber:  delta = 1.345 * MAD / 0.6745  (the 95%-Gaussian-efficiency point)
    #   t_nu:   s     = MAD / t_nu quartile   (= the true scale under t_nu noise)
    #   Cauchy: gamma = MAD                   (Cauchy quartile is 1)
    # MSE and MAE do not need a scale.
    return mad / loss.q_unit

# ----------------------------------------------------------------------------------------------------
# Population quantities: CRB, MCRB, efficiency ratio
# ----------------------------------------------------------------------------------------------------

def sandwich_factor(loss, noise):
    # kappa / lam^2 for the loss with its population scale; N * MCRB of a DC level (X = 1)
    if isinstance(loss, MSE) and not np.isfinite(noise.variance()):
        return np.inf
    c = loss_scale(loss, noise.quartile())
    return loss.kappa(noise, c) / loss.lam(noise, c)**2


def crb_matrix(X, noise):
    return np.linalg.inv(X.T @ X) / noise.fisher()


def mcrb_matrix(X, loss, noise):
    return sandwich_factor(loss, noise) * np.linalg.inv(X.T @ X)


def efficiency_ratio(loss, noise):
    # tr(MCRB) / tr(CRB) = (kappa / lam^2) * I_loc; does not depend on the design X
    return sandwich_factor(loss, noise) * noise.fisher()

# ----------------------------------------------------------------------------------------------------
# Estimators, vectorised over Monte Carlo trials. Y is (R, N), X is (N, d), output (R, d)
# ----------------------------------------------------------------------------------------------------

def _wls(X, Y, W):
    # per-trial weighted least squares: theta_r = (X^T W_r X)^-1 X^T W_r y_r
    G = np.einsum("ni,rn,nj->rij", X, W, X)
    h = np.einsum("ni,rn,rn->ri", X, W, Y)
    return np.linalg.solve(G, h[..., None])[..., 0]


def ols(X, Y):
    return Y @ X @ np.linalg.inv(X.T @ X)


def lad(X, Y, iters=300, tol=1e-10):
    # least absolute deviations by IRLS with weights 1/|r|; checked against the exact LP in the tests
    theta = ols(X, Y)
    for _ in range(iters):
        r = Y - theta @ X.T
        new = _wls(X, Y, 1 / np.maximum(np.abs(r), 1e-9))
        if np.max(np.abs(new - theta)) < tol:
            return new
        theta = new
    return theta


def lad_exact(X, y):
    # LAD for one data set as a linear program: min sum(u + v) s.t. X theta + u - v = y, u, v >= 0
    N, d = X.shape
    cost = np.concatenate([np.zeros(d), np.ones(2 * N)])
    A_eq = np.hstack([X, np.eye(N), -np.eye(N)])
    bounds = [(None, None)] * d + [(0, None)] * (2 * N)
    res = optimize.linprog(cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    return res.x[:d]


def residual_mad(X, Y, theta):
    # robust scale of the residuals: median |r| (residuals of an equivariant fit are centred near 0)
    return np.median(np.abs(Y - theta @ X.T), axis=1)


def m_estimate(X, Y, loss, iters=500, tol=1e-9, return_info=False):
    # M-estimator for any loss in this file.
    # Start: LAD (consistent and equivariant for all noises, unlike OLS under Cauchy noise).
    # Scale: c = loss_scale(MAD of the LAD residuals), held fixed, so the iteration is IRLS on a fixed
    # objective. For the t and Cauchy losses this IRLS is the EM algorithm of Lange et al. and never
    # increases the loss; for Huber it is the usual IRLS. Non-converged trials are reported.
    if isinstance(loss, MSE):
        out = ols(X, Y)
        return (out, np.zeros(len(Y), bool), None) if return_info else out
    theta0 = lad(X, Y)
    if isinstance(loss, MAE):
        return (theta0, np.zeros(len(Y), bool), None) if return_info else theta0
    c = loss_scale(loss, residual_mad(X, Y, theta0))[:, None]
    theta = theta0
    converged = np.zeros(len(Y), bool)
    for _ in range(iters):
        r = Y - theta @ X.T
        new = _wls(X, Y, loss.weight(r, c))
        step = np.max(np.abs(new - theta), axis=1)
        theta = new
        converged = step < tol
        if converged.all():
            break
    return (theta, ~converged, c[:, 0]) if return_info else theta

# ----------------------------------------------------------------------------------------------------
# Sample (data-based) MCRB: the sandwich estimate of Fortunati et al. 2017, Section 4.1
# ----------------------------------------------------------------------------------------------------

def sandwich_cov(X, y, theta, loss, c=1.0, hc="HC0"):
    # A_hat = X^T diag(psi'(r)) X,  B_hat = X^T diag(psi(r)^2) X,  cov = A_hat^-1 B_hat A_hat^-1
    # For MSE this is White's HC0 (or HC3 with hc="HC3"). Not defined for MAE (psi' is a delta).
    r = y - X @ theta
    a = loss.dpsi(r, c)
    b = loss.psi(r, c)**2
    if hc == "HC3":
        h = np.einsum("ni,ij,nj->n", X, np.linalg.inv(X.T @ (a[:, None] * X)), X) * a
        b = b / (1 - h)**2
    Ainv = np.linalg.inv(X.T @ (a[:, None] * X))
    return Ainv @ (X.T @ (b[:, None] * X)) @ Ainv


def design(N, rng, low=-2.0, high=2.0):
    # the design of the plan, Section 4.1: intercept plus x ~ Uniform(-2, 2), drawn once and held fixed
    x = rng.uniform(low, high, N)
    return np.column_stack([np.ones(N), x])
