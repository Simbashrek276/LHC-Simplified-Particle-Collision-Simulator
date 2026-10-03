"""Relativistic kinematics toolkit. Based on section 2.5 of the project report,
docs/report.pdf.

This is the physics engine. It knows nothing about photons or protons by name.
It only works with numbers, the masses in TeV and the four vectors. simulation.py
owns the particle names and calls in here to build the actual energies and momenta.

The main idea is that any final state can be built by splitting things two at a
time. A group of particles is treated as one made up "composite" particle with
its own invariant mass. We split the event into two of those, then split each of
those into its pieces. Every split is a simple two body decay done in the
parent's rest frame, then boosted into the lab.

Read the file in this order.
    1. P4 and its mass
    2. two_body_decay and boost, the two building blocks
    3. draw_composite_mass, picking a composite's mass
    4. two_to_two, two_to_three, two_to_four, putting the blocks together
    5. generate_momenta, the one function simulation.py calls

This is the full 3D version: momentum now has three spatial components
(px, py, pz) instead of two. Random directions are drawn isotropically over the
sphere instead of uniformly over a circle, and the boost is the general 3D
Lorentz boost matrix rather than the 2D special case.
"""

import math
import random
from collections import namedtuple

PI = math.pi

# A composite made of two photons can have a mass of exactly zero. That would
# mean dividing by zero when we split it and a huge boost afterwards. This 1 GeV
# floor keeps the math well behaved. Those events have one very soft photon and
# get thrown out by the energy cut anyway.
MASS_FLOOR = 1e-3  # TeV


# Section 1. The four vector

# A particle's energy and its momentum in 3D (x, y, z) space.
# maybe rename this to NP?
P4 = namedtuple("P4", ["E", "px", "py", "pz"])


def mass(p4):
    """The invariant mass of a four vector.

    We clamp at zero first, because float noise can push a genuinely massless
    particle a tiny bit below zero inside the square root.
    """
    m2 = p4.E ** 2 - p4.px ** 2 - p4.py ** 2 - p4.pz ** 2
    return math.sqrt(max(m2, 0.0))


def invariant_mass(*p4s):
    """The combined mass of several particles treated as one system.

    We add the four vectors up, then take the mass of the total.
    """
    E = sum(p.E for p in p4s)
    px = sum(p.px for p in p4s)
    py = sum(p.py for p in p4s)
    pz = sum(p.pz for p in p4s)
    return mass(P4(E, px, py, pz))


# Section 2. The two building blocks, decay and boost

def two_body_decay(parent_mass, m1, m2):
    """Split a parent that is sitting at rest into two daughters that fly apart.

    Because the parent is at rest, the daughters must fly off in opposite
    directions with equal and opposite momentum. Energy conservation then fixes
    each daughter's energy exactly, so the heavier daughter keeps more of the
    energy. The only random choice is the direction, now drawn isotropically
    over the full sphere (a polar angle theta and an azimuthal angle phi)
    instead of a single angle in a plane. We return the two daughters as four
    vectors in the parent's rest frame.

    On the basis of parent particle staying still
    """
    M = parent_mass  # 13.6 TeV
    E1 = (M ** 2 + m1 ** 2 - m2 ** 2) / (2 * M)
    E2 = M - E1

    # The two daughters share this momentum magnitude. We clamp inside the root
    # so that float noise or a too light parent can never make it negative.
    p = math.sqrt(max(E1 ** 2 - m1 ** 2, 0.0))

    # Isotropic direction on the sphere: phi uniform in [0, 2pi), and
    # cos(theta) uniform in [-1, 1] (NOT theta itself uniform -- that would
    # bunch points near the poles).
    phi = random.uniform(0, 2 * PI)
    cos_theta = random.uniform(-1.0, 1.0)
    sin_theta = math.sqrt(max(1.0 - cos_theta ** 2, 0.0))

    px = p * sin_theta * math.cos(phi)
    py = p * sin_theta * math.sin(phi)
    pz = p * cos_theta

    return P4(E1, px, py, pz), P4(E2, -px, -py, -pz)


def boost(p4, parent_p4):
    """Carry a four vector from a composite's rest frame back into the lab.

    We work out a composite's decay in the frame where it sits still, but the
    composite is really flying through the lab, so its daughters have to be sped
    up to match. parent_p4 is the composite as seen in the lab. p4 is a daughter
    as measured in the composite's rest frame.

    This is now the general 3D Lorentz boost (a 4x4 matrix instead of the
    3x3 special case used when momentum only had two components).

    Input:
        (1) vector p (E, px, py, pz): the energy momentum of the parent particle
        in the lab frame
        (2) vector p4 (E, px, py, pz): the energy momentum of the child particle
        in the rest frame of the parent particle (will be boost's argument)
    Output: vector new_P4(E, px, py, pz): the four vector of the child particle
    in the lab frame (the 3D generalization of equation 8 in docs/report.pdf)
    """
    E_A = parent_p4.E
    if E_A <= 0:
        return p4

    # Boost velocity of the composite, in units where c is 1.
    bx = parent_p4.px / E_A
    by = parent_p4.py / E_A
    bz = parent_p4.pz / E_A
    X = bx ** 2 + by ** 2 + bz ** 2  # |beta|^2

    # If the composite is basically at rest there is nothing to boost.
    if X <= 1e-15:
        return p4

    gamma = 1.0 / math.sqrt(1.0 - X)
    k = (gamma - 1) / X  # shared factor in the spatial-spatial block

    # 4x4 boost matrix, built the same way as the 2D 3x3 version: a time row/
    # column of gamma*beta, and a spatial block of delta_ij + k*beta_i*beta_j.
    A00 = gamma
    A0x, A0y, A0z = gamma * bx, gamma * by, gamma * bz

    Ax0 = gamma * bx
    Axx = k * bx * bx + 1
    Axy = k * bx * by
    Axz = k * bx * bz

    Ay0 = gamma * by
    Ayx = k * by * bx
    Ayy = k * by * by + 1
    Ayz = k * by * bz

    Az0 = gamma * bz
    Azx = k * bz * bx
    Azy = k * bz * by
    Azz = k * bz * bz + 1

    new_E = A00 * p4.E + A0x * p4.px + A0y * p4.py + A0z * p4.pz
    new_px = Ax0 * p4.E + Axx * p4.px + Axy * p4.py + Axz * p4.pz
    new_py = Ay0 * p4.E + Ayx * p4.px + Ayy * p4.py + Ayz * p4.pz
    new_pz = Az0 * p4.E + Azx * p4.px + Azy * p4.py + Azz * p4.pz

    return P4(new_E, new_px, new_py, new_pz)


def decay_in_lab(parent_mass, m1, m2, parent_p4):
    """Split a composite that is moving, and hand back its daughters in the lab.

    This is just the two building blocks one after the other. We decay the
    composite in its own rest frame, then boost each daughter using the
    composite's lab motion. The 2 to 3 and 2 to 4 generators use this so they do
    not repeat the same steps.
    """

    d1, d2 = two_body_decay(parent_mass, m1, m2)  # momentum of a child particle in the resting frame of the parent particle
    return boost(d1, parent_p4), boost(d2, parent_p4)


# Section 3. Picking a composite's mass

def draw_composite_mass(low, high):
    """Roll a random invariant mass for a made up composite particle.

    A composite cannot weigh less than the pieces inside it, which is the low
    limit. The two composites in an event have to share the total energy, which
    is the high limit. We draw uniformly between them, floored so a massless pair
    never lands exactly on zero.
    """
    low = max(low, MASS_FLOOR)
    if high <= low:
        return low
    return random.uniform(low, high)


# Section 4. The generators, one per number of particles that come out.
# These are the 2 to 2, 2 to 3, and 2 to 4 cases from the report (docs/report.pdf).

def two_to_two(total_energy):
    """The collision turns into just two particles. This is the 2 to 2 case.

    With only two of them there is nothing to group, so no composite and no boost
    is needed. Both are massless, so they simply share the energy evenly. Each
    takes half and they fly off in exactly opposite directions. The only thing we
    roll is which way the split points, now isotropically over the sphere.
    """
    E = total_energy / 2.0
    phi = random.uniform(0, 2 * PI)
    cos_theta = random.uniform(-1.0, 1.0)
    sin_theta = math.sqrt(max(1.0 - cos_theta ** 2, 0.0))

    px = E * sin_theta * math.cos(phi)
    py = E * sin_theta * math.sin(phi)
    pz = E * cos_theta
    return [P4(E, px, py, pz), P4(E, -px, -py, -pz)]


def two_to_three(total_energy):
    """The collision turns into three particles. This is the 2 to 3 case.

    We pretend particles 4 and 5 are a single made up particle (45) with its own
    heavy mass. We split the collision into particle 3 plus that composite, then
    crack the composite open into 4 and 5 and boost them into the lab. Particle 3
    and both children are massless. Only the composite (45) carries mass.
    """
    m45 = draw_composite_mass(0.0, total_energy)      # mass of the (45) composite

    p3, p45 = two_body_decay(total_energy, 0.0, m45)  # split collision into 3 and (45)
    p4, p5 = decay_in_lab(m45, 0.0, 0.0, p45)          # split (45) into 4 and 5

    return [p3, p4, p5]


def two_to_four(total_energy):
    """The collision turns into four particles. This is the 2 to 4 case.

    Particles 3 and 4 become one composite (34) and particles 5 and 6 become
    another, (56). We split the collision into those two composites, then crack
    each one open into its pair and boost them into the lab. All four final
    particles are massless. The two composites are what carry the mass.
    """
    # The two composite masses together have to fit inside the total energy. We
    # cap (34) so there is always room left for (56) to exist, then (56) takes a
    # share of whatever is left.
    m34 = draw_composite_mass(0.0, total_energy - MASS_FLOOR)
    m56 = draw_composite_mass(0.0, total_energy - m34)

    p34, p56 = two_body_decay(total_energy, m34, m56)   # split collision into (34) and (56)
    p3, p4 = decay_in_lab(m34, 0.0, 0.0, p34)            # split (34) into 3 and 4
    p5, p6 = decay_in_lab(m56, 0.0, 0.0, p56)            # split (56) into 5 and 6

    return [p3, p4, p5, p6]


# Section 5. The entry point that simulation.py calls.

def generate_momenta(n_particles, total_energy):
    """Build one event and hand back a four vector for every particle.

    All we need to know is how many particles came out. Because every particle is
    massless, their types do not change the kinematics at all, so we just pick the
    matching 2 to 2, 2 to 3, or 2 to 4 generator.
    """
    if n_particles == 2:
        return two_to_two(total_energy)
    if n_particles == 3:
        return two_to_three(total_energy)
    if n_particles == 4:
        return two_to_four(total_energy)
    return None