"""Invariant masses, drawn as outlines on shared axes.

A word on what "the mass of each particle" means here, because it is not what
you might expect. kinematics.py treats every outgoing particle as massless. So
the mass of a single particle is zero by construction, in every process, always.
Plotting it tells you nothing about the physics -- but it is still worth one
figure, because it tells you something about the data file: whatever width that
spike has is pure numerical noise, and it sets the floor on how sharp any other
mass peak in this file can possibly be.

The masses that actually carry information are the ones belonging to groups of
particles. Two massless photons flying apart have a real, heavy combined mass.
That is what the composites in kinematics.py are, and it is what a real detector
reconstructs when it looks for a Higgs. So the rest of this script plots the
invariant mass of every pair.

Which pairs matter depends on how many particles the process has.

    2 to 2   only one pair exists and it is the whole collision, so its mass is
             pinned at 13.6 TeV. Included for completeness.
    2 to 3   particles 4 and 5 came out of a composite, so m(45) is the mass the
             generator drew. m(34) and m(35) are combinations that were never a
             real object, and they are the "background" shape for comparison.
    2 to 4   particles 3 and 4 came out of one composite and 5 and 6 out of the
             other, so m(34) and m(56) are the drawn masses. The four crossed
             pairs get their own figure, since six curves is more than can be
             told apart on one set of axes.

In the p p -> gamma gamma p p process, pair 34 is the two photons, so its
composite-pairs figure is the diphoton mass -- the plot where a Higgs boson
would appear as a bump at 0.125 TeV.

Output, per process folder plots/<process>/:
    mass_single.png
    mass_pairs.png                     (2 to 2 and 2 to 3)
    mass_pairs_composites.png          (2 to 4)
    mass_pairs_crossed.png             (2 to 4)

Run:  python analysis_tools/graph_mass_steps.py
"""

from itertools import combinations
from pathlib import Path

import event_data
import step_plot

# Resolved from this file's location, not the working directory, so the
# script runs the same from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "collision_data" / "events.txt"
OUTPUT_DIR = PROJECT_ROOT / "plots"

TOTAL_ENERGY = 13.6   # TeV
BINS = 50

# Range for the single particle mass figure. Every value in here should be
# numerical noise, so the scale is small: 0.005 TeV is 5 GeV. That is where the
# spread actually lands, because events.txt stores six decimal places and a
# rounding of 1e-6 TeV on E and p works out to a few GeV once it goes through
# m = sqrt(E^2 - p^2) at 13.6 TeV.
NOISE_RANGE = (0.0, 0.005)


def main():
    channels = event_data.load_events(DATA_FILE)

    for state, ch in event_data.iter_processes(channels):
        process = event_data.process_label(state)
        out = OUTPUT_DIR / event_data.process_slug(state)
        out.mkdir(parents=True, exist_ok=True)
        n = ch.n_particles
        print(f"{event_data.process_ascii(state)}: {ch.n_events} events")

        # 1. Single particle masses. Expected to be zero everywhere.
        print("  single particle mass (expected zero, so this measures file precision):")
        step_plot.outline_figure(
            [(ch.label(slot), ch.invariant_mass(slot)) for slot in range(n)],
            bins=BINS,
            value_range=NOISE_RANGE,
            xlabel="Reconstructed single particle mass (TeV)",
            title=f"Single particle mass,   {process}",
            subtitle="every outgoing particle is massless by construction, "
                     "so this width is rounding in events.txt",
            filename=out / "mass_single.png",
            ratio_panel=False,
        )

        # 2. Pair masses. Split into two figures for 2 to 4 processes.
        if n == 4:
            groups = [
                ("composite pairs", [(0, 1), (2, 3)], "mass_pairs_composites.png",
                 "pairs 34 and 56 each came out of one composite, so these are "
                 "the masses the generator drew"),
                ("crossed pairs", [(0, 2), (0, 3), (1, 2), (1, 3)], "mass_pairs_crossed.png",
                 "these pairs were never a single object, so this is the "
                 "combinatorial background shape"),
            ]
        else:
            groups = [("every pair", list(combinations(range(n), 2)), "mass_pairs.png",
                       f"{ch.n_events} events")]

        for group_name, pairs, filename, note in groups:
            print(f"  {group_name}:")
            step_plot.outline_figure(
                [(ch.pair_label(a, b), ch.invariant_mass(a, b)) for a, b in pairs],
                bins=BINS,
                value_range=(0.0, TOTAL_ENERGY),
                xlabel="Invariant mass of the pair (TeV)",
                title=f"Pair invariant mass, {group_name},   {process}",
                subtitle=note,
                filename=out / filename,
            )


if __name__ == "__main__":
    main()
