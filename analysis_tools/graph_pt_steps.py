"""Transverse momentum of every outgoing particle, drawn as outlines.

Transverse momentum is the part of a particle's momentum that points across the
beam rather than along it:

    pt = sqrt(px^2 + py^2) = |p| sin(theta) = E sin(theta)   (massless particles)

It is the single most important quantity at a hadron collider. The incoming
partons carry an unknown share of the protons' momentum along the beam, so the
total momentum along z is never really known -- but across the beam it starts
at zero, so pt is what can actually be balanced and measured.

What the shapes should be:

The 2 to 2 process gives the textbook example of a **Jacobian peak**. Both
particles have E = 6.8 TeV exactly, so pt = 6.8 sin(theta). Because sin(theta)
is flat near theta = 90 degrees, a large range of angles all pile up at nearly
the same pt, and the distribution climbs to a sharp edge at pt = 6.8 TeV. Real
experiments use exactly this edge to measure the masses of W bosons.

For the 2 to 3 and 2 to 4 processes the energies are spread out, so the peak is
smeared into a broad hump.

Output:  plots/<process>/pt.png

Run:  python analysis_tools/graph_pt_steps.py
"""

from pathlib import Path

import event_data
import step_plot

# Resolved from this file's location, not the working directory, so the
# script runs the same from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "collision_data" / "events.txt"
OUTPUT_DIR = PROJECT_ROOT / "plots"

TOTAL_ENERGY = 13.6   # TeV, matches collision.TOTAL_ENERGY
BINS = 50
PT_REFERENCE = 0.1    # TeV, a typical cut; marked as a reference line only


def main():
    channels = event_data.load_events(DATA_FILE)

    for state, ch in event_data.iter_processes(channels):
        process = event_data.process_label(state)
        out = OUTPUT_DIR / event_data.process_slug(state)
        out.mkdir(parents=True, exist_ok=True)
        print(f"{event_data.process_ascii(state)}: {ch.n_events} events")

        if ch.n_particles == 2:
            note = "a Jacobian peak: the curve climbs to a sharp edge at 6.8 TeV"
        else:
            note = "a broad hump, since the energies are spread out"

        pt = ch.pt
        step_plot.outline_figure(
            [(ch.label(slot), pt[:, slot]) for slot in range(ch.n_particles)],
            bins=BINS,
            value_range=(0.0, TOTAL_ENERGY / 2),
            xlabel="Transverse momentum pT = sqrt(px^2 + py^2)  (TeV)",
            title=f"Transverse momentum of each particle,   {process}",
            subtitle=f"{ch.n_events} events, expect {note}; "
                     f"dashed line marks pT = {PT_REFERENCE} TeV",
            filename=out / "pt.png",
            markers=(PT_REFERENCE,),
        )


if __name__ == "__main__":
    main()
