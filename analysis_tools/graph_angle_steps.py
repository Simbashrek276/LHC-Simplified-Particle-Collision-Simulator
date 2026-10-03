"""Emission angles of every outgoing particle, drawn as outlines.

In 3D a particle has two angles, plus two standard ways of re-expressing the
polar one. This script draws all four, once per process:

    theta    the polar angle away from the beam line (z axis), 0 to 180 degrees
    phi      the azimuthal angle around the beam, in the x-y plane, 0 to 360
    cos      cos(theta), which undoes the sphere's geometry (see below)
    eta      pseudorapidity, eta = -ln(tan(theta/2)), the polar angle in the
             units LHC experiments actually use

For theta, each process gets one comparison figure with every particle on it,
plus one figure per particle on its own. A 2 to 4 process therefore gives five
theta figures: four individual and one comparison.

What the shapes should be:

phi should come out flat. There is no preferred direction around the beam.

theta should NOT come out flat. Directions are drawn evenly over the surface of
a sphere, and a sphere has more surface area around its equator than near its
poles, so more particles land near theta = 90. The shape is a sin(theta) arch.

cos(theta) undoes that geometry exactly, so it IS expected to be flat, which
makes it the easiest plot to check the sampling against by eye.

eta should peak at 0 and fall away symmetrically as 1/cosh^2(eta). That is the
same isotropic distribution again, written in eta. The dashed lines mark
|eta| = 2.5, the standard ATLAS/CMS acceptance -- everything outside them would
be lost down the beam pipe.

Output, per process folder plots/<process>/:
    theta.png, theta_particle3.png, theta_particle4.png, ...
    phi.png, costheta.png, eta.png

Run:  python analysis_tools/graph_angle_steps.py
"""

from pathlib import Path

import numpy as np

import event_data
import step_plot

# Resolved from this file's location, not the working directory, so the
# script runs the same from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "collision_data" / "events.txt"
OUTPUT_DIR = PROJECT_ROOT / "plots"

THETA_BINS = 45    # 4 degrees per bin over 0 to 180
PHI_BINS = 45      # 8 degrees per bin over 0 to 360
COS_BINS = 40
ETA_BINS = 50
ETA_RANGE = (-5.0, 5.0)      # holds all but about 0.01% of particles
ETA_ACCEPTANCE = 2.5         # reference lines only; the real cut is in analysis_card.txt


def main():
    channels = event_data.load_events(DATA_FILE)

    for state, ch in event_data.iter_processes(channels):
        process = event_data.process_label(state)
        out = OUTPUT_DIR / event_data.process_slug(state)
        out.mkdir(parents=True, exist_ok=True)
        n = ch.n_particles
        print(f"{event_data.process_ascii(state)}: {ch.n_events} events")

        # theta: one comparison figure with every particle ...
        print("  theta, all particles:")
        step_plot.outline_figure(
            [(ch.label(slot), ch.theta[:, slot]) for slot in range(n)],
            bins=THETA_BINS,
            value_range=(0.0, 180.0),
            xlabel="Polar angle theta from the beam axis (degrees)",
            title=f"Polar angle of each particle,   {process}",
            subtitle=f"{ch.n_events} events, expect a sin(theta) arch peaking at 90 degrees",
            filename=out / "theta.png",
        )

        # ... and one figure per particle on its own.
        for slot in range(n):
            number = ch.particle_number(slot)
            symbol = event_data.particle_symbol(ch.names_in_slot(slot)[0])
            print(f"  theta, particle {number} only:")
            step_plot.outline_figure(
                [(ch.label(slot), ch.theta[:, slot])],
                bins=THETA_BINS,
                value_range=(0.0, 180.0),
                xlabel="Polar angle theta from the beam axis (degrees)",
                title=f"Polar angle of particle {number} (${symbol}$),   {process}",
                subtitle=f"{ch.n_events} events, expect a sin(theta) arch peaking at 90 degrees",
                filename=out / f"theta_particle{number}.png",
            )

        print("  phi:")
        step_plot.outline_figure(
            [(ch.label(slot), ch.phi[:, slot]) for slot in range(n)],
            bins=PHI_BINS,
            value_range=(0.0, 360.0),
            xlabel="Azimuthal angle phi around the beam axis (degrees)",
            title=f"Azimuthal angle of each particle,   {process}",
            subtitle=f"{ch.n_events} events, expect a flat distribution",
            filename=out / "phi.png",
        )

        print("  cos(theta):")
        step_plot.outline_figure(
            [(ch.label(slot), np.cos(np.radians(ch.theta[:, slot]))) for slot in range(n)],
            bins=COS_BINS,
            value_range=(-1.0, 1.0),
            xlabel="cos(theta)",
            title=f"cos of the polar angle,   {process}",
            subtitle=f"{ch.n_events} events, expect a flat distribution if sampling is isotropic",
            filename=out / "costheta.png",
        )

        print("  pseudorapidity:")
        eta = ch.eta
        step_plot.outline_figure(
            [(ch.label(slot), eta[:, slot]) for slot in range(n)],
            bins=ETA_BINS,
            value_range=ETA_RANGE,
            xlabel="Pseudorapidity eta = -ln(tan(theta/2))",
            title=f"Pseudorapidity of each particle,   {process}",
            subtitle=f"{ch.n_events} events, expect a peak at 0 falling as 1/cosh^2(eta); "
                     f"dashed lines mark |eta| = {ETA_ACCEPTANCE}",
            filename=out / "eta.png",
            markers=(-ETA_ACCEPTANCE, ETA_ACCEPTANCE),
        )


if __name__ == "__main__":
    main()
