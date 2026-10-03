"""Tests for the physics engine, utilities/kinematics.py.

These check the properties that make the simulation physically valid: energy
and momentum are conserved, outgoing particles are massless, Lorentz boosts
behave like Lorentz boosts, and random directions really are random.
"""

import math
import random
import unittest

import tests  # noqa: F401  (puts the project root on the import path)
from utilities import kinematics as K

TOTAL_ENERGY = 13.6
TOLERANCE = 1e-6


class TestConservation(unittest.TestCase):
    """Energy and momentum must balance in every event, in every channel."""

    def setUp(self):
        random.seed(12345)

    def test_energy_and_momentum_conserved_in_every_channel(self):
        for n in (2, 3, 4):
            for _ in range(2000):
                particles = K.generate_momenta(n, TOTAL_ENERGY)
                with self.subTest(channel=n):
                    self.assertAlmostEqual(sum(p.E for p in particles), TOTAL_ENERGY,
                                           delta=TOLERANCE)
                    for axis in ("px", "py", "pz"):
                        total = sum(getattr(p, axis) for p in particles)
                        self.assertAlmostEqual(total, 0.0, delta=TOLERANCE)

    def test_every_outgoing_particle_is_massless(self):
        for n in (2, 3, 4):
            for _ in range(2000):
                for p in K.generate_momenta(n, TOTAL_ENERGY):
                    self.assertLess(K.mass(p), 1e-5)

    def test_two_to_two_splits_energy_exactly_in_half(self):
        for _ in range(200):
            a, b = K.two_to_two(TOTAL_ENERGY)
            self.assertEqual(a.E, TOTAL_ENERGY / 2)
            self.assertEqual(b.E, TOTAL_ENERGY / 2)

    def test_two_to_four_composites_fit_inside_the_collision_energy(self):
        # The two composite masses together can never exceed the energy that
        # is available, or the decay momentum would be imaginary.
        for _ in range(20000):
            m34 = K.draw_composite_mass(0.0, TOTAL_ENERGY - K.MASS_FLOOR)
            m56 = K.draw_composite_mass(0.0, TOTAL_ENERGY - m34)
            self.assertLessEqual(m34 + m56, TOTAL_ENERGY + 1e-12)

    def test_unknown_particle_count_returns_none(self):
        self.assertIsNone(K.generate_momenta(5, TOTAL_ENERGY))


class TestTwoBodyDecay(unittest.TestCase):

    def setUp(self):
        random.seed(12345)

    def test_daughters_fly_back_to_back(self):
        d1, d2 = K.two_body_decay(10.0, 2.0, 3.0)
        self.assertAlmostEqual(d1.px, -d2.px)
        self.assertAlmostEqual(d1.py, -d2.py)
        self.assertAlmostEqual(d1.pz, -d2.pz)

    def test_daughter_energies_follow_the_two_body_formula(self):
        # E1 = (M^2 + m1^2 - m2^2) / 2M, so for M=10, m1=2, m2=3 that is 4.75.
        d1, d2 = K.two_body_decay(10.0, 2.0, 3.0)
        self.assertAlmostEqual(d1.E, 4.75)
        self.assertAlmostEqual(d2.E, 5.25)

    def test_daughters_have_the_requested_masses(self):
        d1, d2 = K.two_body_decay(10.0, 2.0, 3.0)
        self.assertAlmostEqual(K.mass(d1), 2.0)
        self.assertAlmostEqual(K.mass(d2), 3.0)


class TestBoost(unittest.TestCase):

    def setUp(self):
        random.seed(12345)

    def test_boost_then_reverse_boost_returns_the_original(self):
        for _ in range(2000):
            parent = K.P4(10.0, random.uniform(-4, 4), random.uniform(-4, 4),
                          random.uniform(-4, 4))
            reverse = K.P4(parent.E, -parent.px, -parent.py, -parent.pz)
            child = K.P4(2.0, 1.0, -0.5, 0.3)
            back = K.boost(K.boost(child, parent), reverse)
            for got, want in zip(back, child):
                self.assertAlmostEqual(got, want, places=9)

    def test_boost_preserves_invariant_mass(self):
        child = K.P4(5.0, 1.0, 2.0, -1.5)
        parent = K.P4(12.0, 3.0, -2.0, 4.0)
        self.assertAlmostEqual(K.mass(K.boost(child, parent)), K.mass(child), places=9)

    def test_parent_at_rest_leaves_the_child_unchanged(self):
        child = K.P4(5.0, 1.0, 2.0, -1.5)
        self.assertEqual(K.boost(child, K.P4(7.0, 0.0, 0.0, 0.0)), child)

    def test_composite_mass_survives_decay_and_boost(self):
        # Split a moving composite in two and add the halves back up: the
        # result must weigh exactly what the composite did.
        for _ in range(2000):
            m45 = K.draw_composite_mass(0.0, TOTAL_ENERGY)
            _, p45 = K.two_body_decay(TOTAL_ENERGY, 0.0, m45)
            p4, p5 = K.decay_in_lab(m45, 0.0, 0.0, p45)
            self.assertAlmostEqual(K.invariant_mass(p4, p5) / m45, 1.0, places=6)


class TestIsotropy(unittest.TestCase):
    """Directions are drawn evenly over the sphere, with no preferred way."""

    SAMPLES = 50000

    def setUp(self):
        random.seed(12345)

    def _directions(self):
        for _ in range(self.SAMPLES):
            p = K.two_to_two(TOTAL_ENERGY)[0]
            magnitude = math.sqrt(p.px ** 2 + p.py ** 2 + p.pz ** 2)
            yield p.pz / magnitude, math.atan2(p.py, p.px) % (2 * math.pi)

    def test_cos_theta_is_flat(self):
        # Flat in cos(theta) is what "evenly over a sphere" means. Each tenth of
        # the range should hold a tenth of the particles, to within 5%.
        bins = [0] * 10
        for cos_theta, _ in self._directions():
            bins[min(9, int((cos_theta + 1) / 2 * 10))] += 1
        expected = self.SAMPLES / 10
        for count in bins:
            self.assertLess(abs(count - expected) / expected, 0.05)

    def test_phi_is_flat(self):
        bins = [0] * 8
        for _, phi in self._directions():
            bins[min(7, int(phi / (2 * math.pi) * 8))] += 1
        expected = self.SAMPLES / 8
        for count in bins:
            self.assertLess(abs(count - expected) / expected, 0.05)


if __name__ == "__main__":
    unittest.main()
