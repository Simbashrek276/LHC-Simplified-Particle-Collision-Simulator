"""LHC collision simulation.

This file runs the experiment. It decides which particles come out of each
collision, then judges the result with a simple detector and logs the events
that pass. All the actual physics, meaning the energies, momenta, conservation,
and boosts, lives in kinematics.py.

The event flow from top to bottom is choose_final_state, then make_event, then
the detector cuts, then write to file.

This is the full 3D version: every particle now carries a py component in
addition to px and pz, and instead of a single planar angle each particle has
two angles -- a polar angle theta (measured from the z axis) and an azimuthal
angle phi (measured around the z axis, in the x-y plane).
"""

import math
import random
from collections import namedtuple
from pathlib import Path

import utilities.kinematics as kinematics

TOTAL_ENERGY = 13.6           # TeV, the LHC collision energy
TARGET_LOGGED_EVENTS = 100000    # stop once this many events pass the detector
OUTPUT_FILE = Path(__file__).resolve().parent / "collision_data" / "events.txt"

# ---------------------------------------------------------------------------
# DETECTOR CUTS -- change these two numbers to change what the detector sees.
# ---------------------------------------------------------------------------
# These are the only two cuts in the system. Both are the standard quantities a
# real LHC experiment cuts on, rather than raw energy and a raw angle.
#
# ETA_MAX is an acceptance in pseudorapidity, eta = -ln(tan(theta/2)). It is the
# polar angle written in the units the field actually uses, and |eta| < 2.5 is
# the usual ATLAS/CMS inner-detector coverage. In plain degrees it means
#
#       9.385 deg  <  theta  <  170.615 deg
#
# so it is exactly a beam-pipe cut: anything closer to the beam than that is
# lost down the hole the beam passes through.
#
# PT_MIN is a cut on transverse momentum, pt = sqrt(px^2 + py^2), the part of
# the momentum that points across the beam rather than along it. A detector
# measures pt, not total energy, and a particle skimming down the beam pipe is
# useless however energetic it is. For a massless particle pt = E * sin(theta),
# so this doubles as an energy cut that is weighted by angle.
ETA_MAX = 2.5                                 # |eta| must be below this
PT_MIN = 0.1                                  # TeV, transverse momentum floor


# One outgoing particle. It carries its name plus everything the detector
# measures about it. theta is the polar angle from the z axis, phi is the
# azimuthal angle around the z axis in the x-y plane. eta and pt are the two
# quantities the cuts are applied to, worked out once in make_event so that
# is_seen does not have to recompute them.
Particle = namedtuple(
    "Particle",
    ["name", "energy", "px", "py", "pz", "theta", "phi", "eta", "pt"],
)


# Step 1. Choose what comes out of the collision.
def choose_final_state():
    """Randomly pick the list of particles produced by one collision."""
    r = random.random()
    if r < 0.25:
        return ["proton", "proton"]
    elif r < 0.43:
        return ["neutron", "antineutron", "proton", "proton"]
    elif r < 0.65:
        return ["photon", "proton", "proton"]
    elif r < 0.80:
        return ["positron", "electron", "proton", "proton"]
    elif r < 0.95:
        return ["muon", "antimuon", "proton", "proton"]
    else:
        return ["photon", "photon", "proton", "proton"]


# Step 2. Turn that list of names into a real event.
def make_event(names):
    """Give each named particle its energy, momentum, and angles.

    We ask the physics engine for the four momenta. It only needs to know how
    many particles there are, since every particle is treated as massless. Then
    we pair each result back with its name and work out its emission angles. We
    return a list of Particle objects.
    """
    momenta = kinematics.generate_momenta(len(names), TOTAL_ENERGY)
    if momenta is None:
        return None

    event = []
    for name, p in zip(names, momenta):
        # theta is measured from the z axis (0 = straight down the beam line,
        # 180 = straight back the other way). phi is measured around the z axis,
        # in the x-y plane, starting from the x axis.
        p_mag = math.sqrt(p.px ** 2 + p.py ** 2 + p.pz ** 2)
        if p_mag > 0:
            theta = math.acos(max(-1.0, min(1.0, p.pz / p_mag)))
        else:
            theta = 0.0
        phi = math.atan2(p.py, p.px) % (2 * math.pi)

        # Transverse momentum: the part of the momentum across the beam.
        pt = math.sqrt(p.px ** 2 + p.py ** 2)

        # Pseudorapidity. We use eta = 0.5*ln((|p|+pz)/(|p|-pz)) rather than
        # -ln(tan(theta/2)) because the two are equal but this form does not
        # blow up when theta is tiny. A particle exactly on the beam line has
        # infinite eta, which correctly fails any finite cut.
        if p_mag > 0 and abs(p.pz) < p_mag:
            eta = 0.5 * math.log((p_mag + p.pz) / (p_mag - p.pz))
        else:
            eta = math.copysign(math.inf, p.pz) if p.pz else 0.0

        event.append(Particle(name, p.E, p.px, p.py, p.pz, theta, phi, eta, pt))
    return event


def collision():
    """Simulate one whole collision. Choose a final state, then generate it."""
    return make_event(choose_final_state())


# Step 3. The detector.
def is_seen(particle):
    """True if the detector can measure this particle, meaning it passes the cuts.

    Two cuts, both set at the top of this file. The particle has to carry enough
    momentum across the beam (PT_MIN), and it has to be far enough away from the
    beam pipe to land inside the detector at all (ETA_MAX).
    """
    if particle.pt < PT_MIN:
        return False
    return abs(particle.eta) < ETA_MAX


def passes_cuts(event):
    """The event is kept only if every particle in it is seen."""
    return all(is_seen(p) for p in event)


def is_conserved(event):
    """Sanity check that energy sums to 13.6 TeV and momentum sums to zero.

    The physics engine guarantees this by construction. We re-check to catch any
    mistake. The small tolerance absorbs the floating point noise that Lorentz
    boosts leave behind.
    """
    total_E = sum(p.energy for p in event)
    total_px = sum(p.px for p in event)
    total_py = sum(p.py for p in event)
    total_pz = sum(p.pz for p in event)
    return (
        math.isclose(total_E, TOTAL_ENERGY, abs_tol=1e-6)
        and math.isclose(total_px, 0.0, abs_tol=1e-6)
        and math.isclose(total_py, 0.0, abs_tol=1e-6)
        and math.isclose(total_pz, 0.0, abs_tol=1e-6)
    )


# Step 4. Write an accepted event to the file.
def write_event(f, event_id, event):
    f.write(f"___EVENT_ID: {event_id}___\n")
    f.write(f"N_Particles: {len(event)}\n")
    f.write(
        f"{'Particle':<15}{'Energy(TeV)':<15}{'Theta(deg)':<15}{'Phi(deg)':<15}"
        f"{'p_x(TeV)':<15}{'p_y(TeV)':<15}{'p_z(TeV)':<15}\n"
    )
    for p in event:
        f.write(
            f"{p.name:<15}{p.energy:<15.6f}{math.degrees(p.theta):<15.6f}"
            f"{math.degrees(p.phi):<15.6f}{p.px:<15.6f}{p.py:<15.6f}{p.pz:<15.6f}\n"
        )
    f.write("\n")


# Step 5. Run collisions until enough events pass.
def run_simulation():
    """Collide until TARGET_LOGGED_EVENTS good events are logged. Returns counts."""
    logged = 0
    total = 0
    # Create collision_data/ if it is not there, so a fresh clone can run this
    # without having to make the folder by hand.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        while logged < TARGET_LOGGED_EVENTS:
            total += 1
            event = collision()

            # Skip anything impossible, not conserving, or unseen by the detector.
            if event is None or not is_conserved(event) or not passes_cuts(event):
                continue

            logged += 1
            write_event(f, logged, event)
    return total, logged


def main():
    total, logged = run_simulation()
    print("Simulation finished.")
    print(f"Total collisions: {total}")
    print(f"Accepted events: {logged}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
