"""LHC collision generator.

This file has one job: produce collisions and write them down. It decides which
particles come out of each collision, asks the physics engine for their energies
and momenta, and logs every event that conserves energy and momentum. The actual
physics lives in utilities/kinematics.py.

It applies NO detector cuts. The file it writes is the complete, unfiltered
sample of what the collisions produced.

    collision.py          generates events        -> collision_data/events.txt
    analysis.py           applies cuts to them    -> reads analysis_card.txt

Splitting it this way means you can try as many different cuts as you like
without regenerating the dataset, which is the slow part. It also matches how a
real experiment works: the detector records everything it can, and the cuts are
a choice made later by whoever is doing the analysis.

The event flow from top to bottom is choose_final_state, then make_event, then
the conservation check, then write to file.

This is the full 3D version: every particle carries a py component in addition
to px and pz, and instead of a single planar angle each particle has two angles
-- a polar angle theta (measured from the z axis) and an azimuthal angle phi
(measured around the z axis, in the x-y plane).
"""

import math
import random
from collections import namedtuple
from pathlib import Path

import utilities.kinematics as kinematics

TOTAL_ENERGY = 13.6           # TeV, the LHC collision energy
TARGET_LOGGED_EVENTS = 100000    # stop once this many events have been written
OUTPUT_FILE = Path(__file__).resolve().parent / "collision_data" / "events.txt"

# A small copy holding just the first few events, written alongside the full
# file. events.txt is far too large for GitHub to display in a browser, so this
# is the one to click on if you just want to see what the data looks like.
SMALL_FILE = OUTPUT_FILE.with_name("events_small.txt")
SMALL_EVENTS = 100

# Random seed. With a number here, every run produces exactly the same events,
# so anyone can reproduce your dataset from this file alone. Any whole number
# works; change it to get a different (but still reproducible) sample. Set it to
# None to get a fresh, unrepeatable sample every time.
#
# One seed covers everything: kinematics.py draws from the same random number
# generator as this file, so seeding here fixes every random choice in the
# whole event, from the final state down to each decay angle.
#
# Note: the events.txt shipped in collision_data/ was generated before this
# setting existed, so that particular file cannot be reproduced exactly. Any
# dataset generated from now on can.
SEED = 2026

# There are deliberately NO detector cuts in this file. Every collision that
# conserves energy and momentum gets written out. Filtering happens later, in
# analysis.py, driven by the numbers in analysis_card.txt. Keeping the two apart
# means you can try a dozen different cuts without regenerating the dataset.

# Real particle rest masses in TeV. Kept for reference only. The simulation now
# treats every outgoing particle as massless, so these are not used when
# generating events. They would matter again if we ever switch back to real masses.
MASS = {
    "photon": 0.0,
    "proton": 0.000938,
    "positron": 0.000000511,
    "electron": 0.000000511,
    "muon": 0.0001057,
    "antimuon": 0.0001057,
    "neutron": 0.000939,
    "antineutron": 0.000939,
}

# One outgoing particle. It carries its name plus everything the detector
# measures about it. theta is the polar angle from the z axis, phi is the
# azimuthal angle around the z axis in the x-y plane.
Particle = namedtuple("Particle", ["name", "energy", "px", "py", "pz", "theta", "phi"])


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

        event.append(Particle(name, p.E, p.px, p.py, p.pz, theta, phi))
    return event


def collision():
    """Simulate one whole collision. Choose a final state, then generate it."""
    return make_event(choose_final_state())


# Step 3. The one check this file does make.
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


# Step 5. Run collisions until enough events are logged.
def run_collisions():
    """Collide until TARGET_LOGGED_EVENTS events are logged. Returns counts.

    Every event that conserves energy and momentum is written out. Nothing is
    filtered here, so the file is the full, unbiased sample. Apply cuts to it
    afterwards with analysis.py.
    """
    if SEED is not None:
        random.seed(SEED)

    logged = 0
    total = 0
    # Create collision_data/ if it is not there, so a fresh clone can run this
    # without having to make the folder by hand.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f, open(SMALL_FILE, "w") as small:
        while logged < TARGET_LOGGED_EVENTS:
            total += 1
            event = collision()

            # The only rejection: something physically impossible.
            if event is None or not is_conserved(event):
                continue

            logged += 1
            write_event(f, logged, event)
            # The first SMALL_EVENTS events also go into the small file, so it
            # is always an exact excerpt of the big one.
            if logged <= SMALL_EVENTS:
                write_event(small, logged, event)
    return total, logged


def main():
    total, logged = run_collisions()
    print("Collision generation finished.")
    print(f"Collisions attempted: {total}")
    print(f"Events written: {logged}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"First {min(SMALL_EVENTS, logged)} events also saved to: {SMALL_FILE}")
    if SEED is None:
        print("Random seed: none (this sample cannot be reproduced exactly)")
    else:
        print(f"Random seed: {SEED} (re-running with the same seed gives identical events)")
    print()
    print("No detector cuts were applied. Run analysis.py to apply the cuts")
    print("listed in analysis_card.txt.")


if __name__ == "__main__":
    main()
