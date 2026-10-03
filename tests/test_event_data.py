"""Tests for the file reader, analysis_tools/event_data.py.

These read collision_data/events_small.txt, the 100-event excerpt, so they run
in a fraction of a second and never need the full 48 MB file.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests  # noqa: F401  (puts the project root on the import path)
from analysis_tools import event_data

SMALL_FILE = event_data.DATA_DIR / "events_small.txt"

# events.txt stores six decimal places, so values read back are only good to
# about 1e-6 each. Sums of four of them can be off by a few times that.
FILE_ROUNDING = 1e-5


class TestReading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channels = event_data.load_events(SMALL_FILE)

    def test_reads_all_one_hundred_events(self):
        self.assertEqual(sum(ch.n_events for ch in self.channels.values()), 100)

    def test_only_two_three_and_four_particle_channels(self):
        self.assertEqual(set(self.channels), {2, 3, 4})

    def test_every_event_read_back_still_conserves_energy_and_momentum(self):
        for n, ch in self.channels.items():
            with self.subTest(channel=n):
                np.testing.assert_allclose(ch.energy.sum(axis=1), 13.6, atol=FILE_ROUNDING)
                for axis in (ch.px, ch.py, ch.pz):
                    np.testing.assert_allclose(axis.sum(axis=1), 0.0, atol=FILE_ROUNDING)

    def test_transverse_momentum_equals_energy_times_sin_theta(self):
        for ch in self.channels.values():
            expected = ch.energy * np.sin(np.radians(ch.theta))
            np.testing.assert_allclose(ch.pt, expected, atol=1e-4)

    def test_pseudorapidity_equals_minus_log_tan_half_theta(self):
        for ch in self.channels.values():
            expected = -np.log(np.tan(np.radians(ch.theta) / 2))
            np.testing.assert_allclose(ch.eta, expected, atol=1e-3)

    def test_a_half_written_final_event_is_ignored(self):
        text = SMALL_FILE.read_text()
        # Chop the file part way through its last event.
        truncated = text[: text.rfind("___EVENT_ID")] + \
            "___EVENT_ID: 101___\nN_Particles: 4\nParticle ...\nproton 1 2 3 4 5 6\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truncated.txt"
            path.write_text(truncated)
            channels = event_data.load_events(path)
        self.assertEqual(sum(ch.n_events for ch in channels.values()), 99)


class TestProcesses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.channels = event_data.load_events(SMALL_FILE)
        cls.processes = list(event_data.iter_processes(cls.channels))

    def test_processes_add_up_to_every_event(self):
        self.assertEqual(sum(sub.n_events for _, sub in self.processes), 100)

    def test_each_process_holds_exactly_one_final_state(self):
        for state, sub in self.processes:
            self.assertEqual(sub.state, state)

    def test_process_names(self):
        state = ("positron", "electron", "proton", "proton")
        self.assertEqual(event_data.process_ascii(state), "p p -> e+ e- p p")
        self.assertEqual(event_data.process_slug(state), "pp_to_eplus_eminus_p_p")
        self.assertIn(r"\rightarrow", event_data.process_label(state))


if __name__ == "__main__":
    unittest.main()
