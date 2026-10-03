"""Tests for the event generator, collision.py.

None of these write to collision_data/. Every test that runs the generator
points its output at a temporary folder first and restores the real paths
afterwards, so running the tests can never overwrite your dataset.
"""

import math
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import tests  # noqa: F401  (puts the project root on the import path)
import collision


class GeneratorOutputTest(unittest.TestCase):
    """Base class: redirect collision.py's output files to a temporary folder."""

    def setUp(self):
        self._saved = {name: getattr(collision, name) for name in
                       ("OUTPUT_FILE", "SMALL_FILE", "TARGET_LOGGED_EVENTS", "SEED")}
        self._tmp = tempfile.TemporaryDirectory()
        folder = Path(self._tmp.name)
        collision.OUTPUT_FILE = folder / "events.txt"
        collision.SMALL_FILE = folder / "events_small.txt"
        collision.TARGET_LOGGED_EVENTS = 250

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(collision, name, value)
        self._tmp.cleanup()


class TestEvents(unittest.TestCase):

    def setUp(self):
        random.seed(12345)

    def test_angles_rebuild_the_momentum(self):
        # theta and phi are worked out from px, py, pz. Going back the other way
        # must give the same momentum.
        for _ in range(2000):
            for p in collision.collision():
                magnitude = math.sqrt(p.px ** 2 + p.py ** 2 + p.pz ** 2)
                self.assertAlmostEqual(magnitude * math.sin(p.theta) * math.cos(p.phi), p.px)
                self.assertAlmostEqual(magnitude * math.sin(p.theta) * math.sin(p.phi), p.py)
                self.assertAlmostEqual(magnitude * math.cos(p.theta), p.pz)

    def test_every_generated_event_passes_the_conservation_check(self):
        for _ in range(2000):
            self.assertTrue(collision.is_conserved(collision.collision()))

    def test_conservation_check_catches_a_broken_event(self):
        event = collision.collision()
        broken = [event[0]._replace(energy=event[0].energy + 0.5)] + event[1:]
        self.assertFalse(collision.is_conserved(broken))

    def test_final_states_occur_at_the_advertised_rates(self):
        draws = 100000
        counts = Counter(tuple(collision.choose_final_state()) for _ in range(draws))
        expected = {
            ("proton", "proton"): 0.25,
            ("photon", "proton", "proton"): 0.22,
            ("neutron", "antineutron", "proton", "proton"): 0.18,
            ("positron", "electron", "proton", "proton"): 0.15,
            ("muon", "antimuon", "proton", "proton"): 0.15,
            ("photon", "photon", "proton", "proton"): 0.05,
        }
        self.assertEqual(set(counts), set(expected))
        for state, rate in expected.items():
            self.assertAlmostEqual(counts[state] / draws, rate, delta=0.01)

    def test_generator_applies_no_detector_cuts(self):
        # Cuts belong in analysis.py. This guards against them creeping back in.
        for name in ("is_seen", "passes_cuts", "ETA_MAX", "PT_MIN", "MIN_ENERGY"):
            self.assertFalse(hasattr(collision, name),
                             f"collision.py should not define {name}")


class TestOutputFiles(GeneratorOutputTest):

    def test_same_seed_gives_identical_files(self):
        collision.SEED = 7
        collision.run_collisions()
        first = collision.OUTPUT_FILE.read_text()
        collision.run_collisions()
        second = collision.OUTPUT_FILE.read_text()
        self.assertEqual(first, second)

    def test_different_seeds_give_different_files(self):
        collision.SEED = 7
        collision.run_collisions()
        first = collision.OUTPUT_FILE.read_text()
        collision.SEED = 8
        collision.run_collisions()
        self.assertNotEqual(first, collision.OUTPUT_FILE.read_text())

    def test_writes_exactly_the_requested_number_of_events(self):
        collision.SEED = 7
        total, logged = collision.run_collisions()
        self.assertEqual(logged, 250)
        self.assertEqual(collision.OUTPUT_FILE.read_text().count("___EVENT_ID"), 250)

    def test_small_file_is_an_exact_excerpt_of_the_big_one(self):
        collision.SEED = 7
        collision.run_collisions()
        big = collision.OUTPUT_FILE.read_text()
        small = collision.SMALL_FILE.read_text()
        self.assertEqual(small.count("___EVENT_ID"), collision.SMALL_EVENTS)
        self.assertTrue(big.startswith(small))


if __name__ == "__main__":
    unittest.main()
