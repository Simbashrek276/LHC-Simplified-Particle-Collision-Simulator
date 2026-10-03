"""Tests for the cut stage, analysis.py and its analysis_card.txt.

Cards are written to a temporary folder for each test, so the real
analysis_card.txt is never touched.
"""

import math
import tempfile
import unittest
from pathlib import Path

import tests  # noqa: F401  (puts the project root on the import path)
import analysis
from analysis_tools import event_data

SMALL_FILE = event_data.DATA_DIR / "events_small.txt"

ALL_CUTS_OFF = """
min_energy = 0
theta_min_deg = 0
theta_max_deg = 180
pseudo_rapidity_max = none
transverse_momentum_min = 0
"""


class CardTest(unittest.TestCase):
    """Base class: write a card to a temporary file and read it back."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def card(self, text):
        path = Path(self._tmp.name) / "card.txt"
        path.write_text(text)
        return analysis.read_card(path)


class TestReadingTheCard(CardTest):

    def test_the_shipped_card_is_valid(self):
        settings = analysis.read_card(analysis.DEFAULT_CARD)
        self.assertEqual(set(settings), set(analysis.DEFAULTS))

    def test_values_are_read_as_numbers(self):
        settings = self.card("min_energy = 0.5\ntransverse_momentum_min = 0.1\n")
        self.assertEqual(settings["min_energy"], 0.5)
        self.assertEqual(settings["transverse_momentum_min"], 0.1)

    def test_comments_and_blank_lines_are_ignored(self):
        settings = self.card("# a comment\n\nmin_energy = 0.5   # trailing comment\n")
        self.assertEqual(settings["min_energy"], 0.5)

    def test_none_switches_the_pseudorapidity_cut_off(self):
        self.assertIsNone(self.card("pseudo_rapidity_max = none\n")["pseudo_rapidity_max"])

    def test_missing_settings_fall_back_to_the_defaults(self):
        self.assertEqual(self.card("min_energy = 0.5\n")["theta_max_deg"], 180.0)


class TestBadCardsAreRejected(CardTest):
    """A cut that silently fails to apply is the worst outcome, so every one of
    these has to stop the program rather than carry on."""

    def assertRejected(self, text):
        with self.assertRaises(SystemExit):
            self.card(text)

    def test_misspelled_setting(self):
        self.assertRejected("min_enrgy = 0.5\n")

    def test_value_that_is_not_a_number(self):
        self.assertRejected("min_energy = abc\n")

    def test_line_without_an_equals_sign(self):
        self.assertRejected("min_energy 0.5\n")

    def test_theta_window_upside_down(self):
        self.assertRejected("theta_min_deg = 150\ntheta_max_deg = 30\n")

    def test_theta_outside_zero_to_180(self):
        self.assertRejected("theta_max_deg = 200\n")

    def test_negative_energy(self):
        self.assertRejected("min_energy = -1\n")

    def test_yes_no_setting_given_something_else(self):
        self.assertRejected("require_all_particles = maybe\n")


class TestSingleParticleCuts(CardTest):

    def test_energy_cut(self):
        s = self.card(ALL_CUTS_OFF + "min_energy = 1.0\n")
        self.assertFalse(analysis.particle_passes(0.5, 90, 0.5, 0, 0, s))
        self.assertTrue(analysis.particle_passes(1.5, 90, 1.5, 0, 0, s))

    def test_theta_window(self):
        s = self.card(ALL_CUTS_OFF + "theta_min_deg = 30\ntheta_max_deg = 150\n")
        self.assertFalse(analysis.particle_passes(1.0, 10, 0.17, 0, 0.98, s))
        self.assertTrue(analysis.particle_passes(1.0, 90, 1.0, 0, 0, s))

    def test_transverse_momentum_cut(self):
        s = self.card(ALL_CUTS_OFF + "transverse_momentum_min = 0.1\n")
        # Nearly all of this particle's momentum is along the beam, so pT < 0.1.
        self.assertFalse(analysis.particle_passes(1.0, 3, 0.05, 0, 0.9987, s))
        self.assertTrue(analysis.particle_passes(1.0, 90, 1.0, 0, 0, s))

    def test_pseudorapidity_cut_matches_its_angle(self):
        s = self.card(ALL_CUTS_OFF + "pseudo_rapidity_max = 2.5\n")
        edge = 2 * math.degrees(math.atan(math.exp(-2.5)))   # 9.385 degrees
        for theta, should_pass in ((edge - 0.5, False), (edge + 0.5, True), (90, True)):
            t = math.radians(theta)
            passed = analysis.particle_passes(1.0, theta, math.sin(t), 0, math.cos(t), s)
            self.assertEqual(passed, should_pass, f"theta = {theta:.3f}")


class TestWholeEvents(CardTest):

    def test_all_cuts_off_keeps_every_event(self):
        results = analysis.analyse(self.card(ALL_CUTS_OFF), SMALL_FILE)
        self.assertEqual(sum(r["n_kept"] for r in results.values()), 100)

    def test_every_kept_particle_really_passes_the_cuts(self):
        s = self.card(ALL_CUTS_OFF + "pseudo_rapidity_max = 2.5\ntransverse_momentum_min = 0.1\n")
        results = analysis.analyse(s, SMALL_FILE)
        for n, r in results.items():
            ch = r["channel"]
            for i, kept in enumerate(r["mask"]):
                if kept:
                    self.assertTrue((ch.pt[i] >= 0.1).all())
                    self.assertTrue((abs(ch.eta[i]) < 2.5).all())

    def test_requiring_one_particle_keeps_at_least_as_many_as_requiring_all(self):
        cuts = ALL_CUTS_OFF + "transverse_momentum_min = 1.0\n"
        strict = analysis.analyse(self.card(cuts + "require_all_particles = yes\n"), SMALL_FILE)
        loose = analysis.analyse(self.card(cuts + "require_all_particles = no\n"), SMALL_FILE)
        for n in strict:
            self.assertGreaterEqual(loose[n]["n_kept"], strict[n]["n_kept"])


if __name__ == "__main__":
    unittest.main()
