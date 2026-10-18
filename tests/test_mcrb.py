# Author: Bhuvi
# Tests for Classic_sim/mcrb.py. Run from the repository root with:
#   python -m unittest discover -s tests -v
# Only the standard library unittest is used, so no extra package is needed.

import os
import sys
import unittest
import numpy as np
from scipy import stats, integrate, optimize, special

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Classic_sim"))
import mcrb as m


class TestFisherInformation(unittest.TestCase):
    # the closed forms of Sections 3 and 6 against direct quadrature of E[u(w)^2]

    def test_closed_forms_match_quadrature(self):
        for noise in [m.Gaussian(1.0), m.Gaussian(2.0), m.Laplace(1.0), m.Laplace(0.5),
                      m.StudentT(3), m.StudentT(5, 2.0), m.StudentT(1)]:
            numeric = m.Noise.fisher(noise)  # generic quadrature, bypassing the closed form
            self.assertAlmostEqual(numeric, noise.fisher(), places=6, msg=noise.name)

    def test_contaminated_matches_independent_integral(self):
        # sim_H computes the same quantity as the integral of p'(w)^2 / p(w)
        eps, s0, s1 = 0.1, 1.0, 5.0
        def integrand(w):
            p0, p1 = stats.norm.pdf(w, 0, s0), stats.norm.pdf(w, 0, s1)
            p = (1 - eps) * p0 + eps * p1
            dp = -(1 - eps) * w / s0**2 * p0 - eps * w / s1**2 * p1
            return dp**2 / p
        ref = integrate.quad(integrand, -60, 60, limit=400, points=[-5, -1, 0, 1, 5])[0]
        self.assertAlmostEqual(m.Contaminated(eps, s0, s1).fisher(), ref, places=6)

    def test_contaminated_score_has_no_nan_in_tails(self):
        w = np.array([-1e3, -40.0, 0.0, 40.0, 1e3])
        self.assertTrue(np.all(np.isfinite(m.Contaminated().score(w))))


class TestPopulationSandwich(unittest.TestCase):

    def test_three_gates_and_diagonal(self):
        # plan Section 4.2: diagonal = 1, MSE under Laplace = 2, MSE under t3 = 2
        cases = [(m.MSE(), m.Gaussian(), 1.0), (m.MAE(), m.Laplace(), 1.0),
                 (m.TLoss(3), m.StudentT(3), 1.0), (m.TLoss(1), m.StudentT(1), 1.0),
                 (m.MSE(), m.Laplace(), 2.0), (m.MSE(), m.StudentT(3), 2.0)]
        for loss, noise, target in cases:
            self.assertAlmostEqual(m.efficiency_ratio(loss, noise), target, places=4,
                                   msg="%s under %s" % (loss.name, noise.name))

    def test_known_values(self):
        self.assertAlmostEqual(m.efficiency_ratio(m.MAE(), m.Gaussian()), np.pi / 2, places=6)
        # Huber with k = 1.345 has 95% efficiency at the Gaussian
        self.assertAlmostEqual(1 / m.efficiency_ratio(m.Huber(1.345), m.Gaussian()), 0.950, places=3)
        # median under t3: (nu+1) / (4 (nu+3) p(0)^2) = pi^2/8, report Section 6.3
        self.assertAlmostEqual(m.efficiency_ratio(m.MAE(), m.StudentT(3)), np.pi**2 / 8, places=6)
        # mean under t_nu: nu (nu+1) / ((nu-2)(nu+3))
        nu = 5
        self.assertAlmostEqual(m.efficiency_ratio(m.MSE(), m.StudentT(nu)),
                               nu * (nu + 1) / ((nu - 2) * (nu + 3)), places=6)
        self.assertTrue(np.isinf(m.efficiency_ratio(m.MSE(), m.StudentT(1))))

    def test_mcrb_never_below_crb(self):
        # Cauchy-Schwarz: kappa / lam^2 >= 1 / I_loc for every loss and noise
        for loss in m.default_losses():
            for noise in m.default_noises():
                self.assertGreaterEqual(m.efficiency_ratio(loss, noise), 1 - 1e-6,
                                        msg="%s under %s" % (loss.name, noise.name))

    def test_ratio_is_scale_invariant(self):
        # with the MAD scale rule the ratio does not depend on the noise scale
        for loss in m.default_losses():
            r1 = m.efficiency_ratio(loss, m.Laplace(1.0))
            r2 = m.efficiency_ratio(loss, m.Laplace(3.0))
            self.assertAlmostEqual(r1, r2, places=5, msg=loss.name)

    def test_regression_matrix_form(self):
        rng = np.random.default_rng(0)
        X = m.design(50, rng)
        loss, noise = m.Huber(), m.Laplace()
        M = m.mcrb_matrix(X, loss, noise)
        C = m.crb_matrix(X, noise)
        self.assertTrue(np.allclose(M, m.sandwich_factor(loss, noise) * np.linalg.inv(X.T @ X)))
        self.assertAlmostEqual(np.trace(M) / np.trace(C), m.efficiency_ratio(loss, noise), places=10)


class TestEstimators(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.X = m.design(60, self.rng)
        self.theta = np.array([1.0, 2.0])

    def data(self, noise, R=5):
        return self.theta @ self.X.T + noise.sample(self.rng, (R, len(self.X)))

    def test_ols_closed_form(self):
        Y = self.data(m.Gaussian())
        ref = np.array([np.linalg.lstsq(self.X, y, rcond=None)[0] for y in Y])
        self.assertTrue(np.allclose(m.ols(self.X, Y), ref))

    def test_lad_irls_matches_linear_program(self):
        Y = self.data(m.Laplace(), R=8)
        irls = m.lad(self.X, Y)
        for r in range(len(Y)):
            exact = m.lad_exact(self.X, Y[r])
            f = lambda t: np.sum(np.abs(Y[r] - self.X @ t))
            # same objective value to high accuracy (the LAD minimiser can be non-unique)
            self.assertLess(f(irls[r]) - f(exact), 1e-6 * f(exact))
            self.assertTrue(np.allclose(irls[r], exact, atol=1e-4))

    def test_m_estimates_minimise_their_loss(self):
        for loss, noise in [(m.Huber(), m.Laplace()), (m.TLoss(3), m.StudentT(3)), (m.TLoss(1), m.StudentT(1))]:
            Y = self.data(noise, R=4)
            est, failed, c = m.m_estimate(self.X, Y, loss, return_info=True)
            self.assertFalse(failed.any())
            for r in range(len(Y)):
                obj = lambda t: np.sum(loss.rho(Y[r] - self.X @ t, c[r]))
                ref = optimize.minimize(obj, est[r] + 0.05, method="Nelder-Mead",
                                        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000}).x
                self.assertLessEqual(obj(est[r]), obj(ref) + 1e-8, msg=loss.name)
                # gradient of the objective is zero at the estimate
                grad = -self.X.T @ loss.psi(Y[r] - self.X @ est[r], c[r])
                self.assertLess(np.max(np.abs(grad)), 1e-6, msg=loss.name)

    def test_regression_equivariance(self):
        Y = self.data(m.StudentT(3), R=3)
        shift = np.array([0.7, -1.3])
        for loss in m.default_losses():
            a = m.m_estimate(self.X, Y, loss)
            b = m.m_estimate(self.X, Y + shift @ self.X.T, loss)
            self.assertTrue(np.allclose(b - a, shift, atol=1e-6), msg=loss.name)

    def test_sandwich_for_mse_is_white_hc0(self):
        y = self.data(m.Gaussian(), R=1)[0]
        th = np.linalg.lstsq(self.X, y, rcond=None)[0]
        r = y - self.X @ th
        XtXi = np.linalg.inv(self.X.T @ self.X)
        hc0 = XtXi @ self.X.T @ np.diag(r**2) @ self.X @ XtXi
        self.assertTrue(np.allclose(m.sandwich_cov(self.X, y, th, m.MSE()), hc0))

    def test_sample_sandwich_is_consistent(self):
        # at large N the data-based sandwich is close to the population MCRB
        rng = np.random.default_rng(5)
        X = m.design(20000, rng)
        loss, noise = m.Huber(), m.Laplace()
        y = self.theta @ X.T + noise.sample(rng, len(X))
        est, _, c = m.m_estimate(X, y[None, :], loss, return_info=True)
        S = m.sandwich_cov(X, y, est[0], loss, c[0])
        M = m.mcrb_matrix(X, loss, noise)
        self.assertLess(abs(np.trace(S) / np.trace(M) - 1), 0.05)


if __name__ == "__main__":
    unittest.main()
