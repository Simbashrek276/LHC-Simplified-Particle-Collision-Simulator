"""Energy of every outgoing particle, drawn as outlines on shared axes.

One figure per process, for example  p p -> e+ e- p p.  Every particle in that
process gets a curve, so you can see straight away which particle carries more
of the 13.6 TeV.

Output:  plots/<process>/energy.png

Run:  python analysis_tools/graph_energy_steps.py
"""

from pathlib import Path

import event_data
import step_plot

# Resolved from this file's location, not the working directory, so the
# script runs the same from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "collision_data" / "events.txt"
OUTPUT_DIR = PROJECT_ROOT / "plots"

TOTAL_ENERGY = 13.6   # TeV, matches simulation.TOTAL_ENERGY
BINS = 50


def main():
    channels = event_data.load_events(DATA_FILE)

    for state, ch in event_data.iter_processes(channels):
        process = event_data.process_label(state)
        out = OUTPUT_DIR / event_data.process_slug(state)
        out.mkdir(parents=True, exist_ok=True)
        print(f"{event_data.process_ascii(state)}: {ch.n_events} events")

        step_plot.outline_figure(
            [(ch.label(slot), ch.energy[:, slot]) for slot in range(ch.n_particles)],
            bins=BINS,
            value_range=(0.0, TOTAL_ENERGY / 2),
            xlabel="Energy (TeV)",
            title=f"Energy of each particle,   {process}",
            subtitle=f"{ch.n_events} events, {TOTAL_ENERGY} TeV total",
            filename=out / "energy.png",
        )

        # The 2 to 2 case has both particles pinned to exactly half the
        # collision energy, so its two curves are a single spike sitting on top
        # of each other. That is correct, not a drawing bug -- with two massless
        # particles and nothing else, energy conservation leaves no freedom.
        if ch.n_particles == 2:
            print("    note: both curves are one spike at 6.8 TeV, they overlap exactly")


if __name__ == "__main__":
    main()
