"""Shared reader for events.txt. Used by analysis.py and the graphing scripts.

The old graphing scripts each re-implemented their own line parser and each one
hard coded a column number. That broke when the simulation went to 3D, because
the row layout gained two columns. This module owns the layout in one place so a
future format change is a one line fix.

The 3D row layout written by collision.write_event is

    token  0        1            2           3          4         5         6
           name     Energy(TeV)  Theta(deg)  Phi(deg)   p_x(TeV)  p_y(TeV)  p_z(TeV)

theta is the polar angle from the beam (z) axis and runs 0 to 180 degrees.
phi is the azimuthal angle around the beam, in the x-y plane, 0 to 360 degrees.

Events are grouped two ways.

    channel   how many particles came out (2, 3 or 4). This is what picks the
              generator in kinematics.py.
    process   the exact final state, such as p p -> e+ e- p p. One channel can
              hold several processes: the 2 to 4 channel holds four of them.

The graphing scripts work per process, because a figure titled "2 to 4" would be
mixing muons, electrons, neutrons and photons on the same curve.
"""

from collections import Counter
from pathlib import Path

import numpy as np

# Paths are worked out from this file's own location rather than from the
# working directory, so the scripts behave the same whether you run them from
# the project root, from inside analysis_tools/, or from the VS Code Run button.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "collision_data"
DEFAULT_FILE = DATA_DIR / "events.txt"

# Slot index 0 is "particle 3", because particles 1 and 2 are the incoming beam
# protons and are not written to the file.
FIRST_PARTICLE_NUMBER = 3

# How each particle name is written. The first form is matplotlib mathtext, for
# figure titles and legends. The second is plain ASCII, for folder names and for
# printing to a terminal that may not cope with Greek letters.
PARTICLE_SYMBOLS = {
    "proton":      (r"p",        "p"),
    "neutron":     (r"n",        "n"),
    "antineutron": (r"\bar{n}",  "nbar"),
    "photon":      (r"\gamma",   "gamma"),
    "positron":    (r"e^{+}",    "eplus"),
    "electron":    (r"e^{-}",    "eminus"),
    "muon":        (r"\mu^{-}",  "muminus"),
    "antimuon":    (r"\mu^{+}",  "muplus"),
}


def particle_symbol(name):
    """Mathtext for one particle, without the surrounding $ signs."""
    return PARTICLE_SYMBOLS.get(name, (name, name))[0]


def particle_ascii(name):
    """Plain ASCII name for one particle."""
    return PARTICLE_SYMBOLS.get(name, (name, name))[1]


def process_label(state):
    """The process as a figure title, e.g.  $p\\,p \\rightarrow e^{+}\\,e^{-}\\,p\\,p$.

    The two incoming beam protons are always written first, since every process
    in this simulation starts from a proton-proton collision.
    """
    final = r"\,".join(particle_symbol(n) for n in state)
    return rf"$p\,p \rightarrow {final}$"


def process_ascii(state):
    """The process in plain text, e.g.  p p -> e+ e- p p."""
    short = {"eplus": "e+", "eminus": "e-", "muminus": "mu-", "muplus": "mu+"}
    final = " ".join(short.get(particle_ascii(n), particle_ascii(n)) for n in state)
    return f"p p -> {final}"


def process_slug(state):
    """A folder name for the process, e.g.  pp_to_eplus_eminus_p_p."""
    return "pp_to_" + "_".join(particle_ascii(n) for n in state)


class Channel:
    """Every logged event that produced the same number of particles.

    All the arrays have shape (n_events, n_particles). Column j is always the
    j-th particle row as written in the file, so column 0 is particle 3.
    """

    def __init__(self, n_particles, rows, states):
        self.n_particles = n_particles
        self.states = states                      # one name tuple per event

        block = np.asarray(rows, dtype=float).reshape(-1, n_particles, 6)
        self.energy = block[:, :, 0]
        self.theta = block[:, :, 1]               # degrees, 0 to 180
        self.phi = block[:, :, 2]                 # degrees, 0 to 360
        self.px = block[:, :, 3]
        self.py = block[:, :, 4]
        self.pz = block[:, :, 5]

    @property
    def n_events(self):
        return self.energy.shape[0]

    @property
    def pt(self):
        """Transverse momentum, the part of the momentum across the beam (TeV).

        pt = sqrt(px^2 + py^2) = |p| sin(theta).
        """
        return np.sqrt(self.px ** 2 + self.py ** 2)

    @property
    def eta(self):
        """Pseudorapidity, eta = -ln(tan(theta/2)).

        Computed as 0.5*ln((|p|+pz)/(|p|-pz)), which is the same number but does
        not lose precision when theta is tiny. A particle exactly on the beam
        line would have infinite eta; with 100,000 events that never happens.
        """
        p = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.5 * np.log((p + self.pz) / (p - self.pz))

    def particle_number(self, slot):
        """File slot index to the particle number used in docs/report.pdf (3, 4, 5, 6)."""
        return slot + FIRST_PARTICLE_NUMBER

    def names_in_slot(self, slot):
        """Every particle name that has appeared in this slot, most common first.

        A slot is not always the same particle. The 2 to 4 channel mixes several
        final states, so slot 0 is sometimes a muon and sometimes a photon. That
        does not change the kinematics, because the generator treats every
        outgoing particle as massless, but it does change what a sensible label
        for the curve is.
        """
        counts = Counter(state[slot] for state in self.states)
        return [name for name, _ in counts.most_common()]

    def label(self, slot):
        """A legend label like 'Particle 3: $e^{+}$', or '(mixed)' if it varies."""
        names = self.names_in_slot(slot)
        if len(names) == 1:
            return f"Particle {self.particle_number(slot)}: ${particle_symbol(names[0])}$"
        return f"Particle {self.particle_number(slot)} (mixed)"

    def pair_label(self, a, b):
        """A legend label for a pair, like 'Pair 34: $e^{+}e^{-}$'."""
        tag = f"Pair {self.particle_number(a)}{self.particle_number(b)}"
        names_a, names_b = self.names_in_slot(a), self.names_in_slot(b)
        if len(names_a) == 1 and len(names_b) == 1:
            return f"{tag}: ${particle_symbol(names_a[0])}\\,{particle_symbol(names_b[0])}$"
        return tag

    @property
    def state(self):
        """The single final state of this Channel, if it holds only one."""
        distinct = set(self.states)
        return next(iter(distinct)) if len(distinct) == 1 else None

    def four_vector(self, slot):
        """(E, px, py, pz) arrays for one particle slot, one entry per event."""
        return (
            self.energy[:, slot],
            self.px[:, slot],
            self.py[:, slot],
            self.pz[:, slot],
        )

    def invariant_mass(self, *slots):
        """Invariant mass of a group of particles, one value per event.

        Adds the four vectors of the chosen slots together and takes the mass of
        the total. Clamped at zero first, because floating point noise can push
        a genuinely massless combination a hair below zero inside the root. This
        is the 3D form, so p_y is included -- the old 2D scripts left it out.
        """
        E = np.zeros(self.n_events)
        px = np.zeros(self.n_events)
        py = np.zeros(self.n_events)
        pz = np.zeros(self.n_events)
        for slot in slots:
            E += self.energy[:, slot]
            px += self.px[:, slot]
            py += self.py[:, slot]
            pz += self.pz[:, slot]
        m2 = E ** 2 - px ** 2 - py ** 2 - pz ** 2
        return np.sqrt(np.clip(m2, 0.0, None))

    def filter_state(self, state):
        """A new Channel holding only the events with this exact final state.

        Pass a tuple of names, for example ("photon", "photon", "proton",
        "proton"). Useful for pulling the diphoton events out of channel 4.
        """
        state = tuple(state)
        keep = np.array([s == state for s in self.states])
        if not keep.any():
            raise ValueError(f"no events in channel {self.n_particles} with state {state}")

        # Slice the arrays directly rather than rebuilding from a list of rows,
        # which is about ten times faster on the 2 to 4 channel.
        sub = Channel.__new__(Channel)
        sub.n_particles = self.n_particles
        sub.states = [s for s, k in zip(self.states, keep) if k]
        for name in ("energy", "theta", "phi", "px", "py", "pz"):
            setattr(sub, name, getattr(self, name)[keep])
        return sub


def load_events(filename=DEFAULT_FILE):
    """Read the whole events file. Returns {n_particles: Channel}.

    Anything that is not a particle row is skipped by trying to read its six
    numbers and moving on if that fails, so the column header line is filtered
    out without having to match its exact text.
    """
    rows = {}
    states = {}

    n_expected = 0
    event_rows = []
    event_names = []

    def flush():
        # Only keep an event whose row count matches its own N_Particles line,
        # so a half written final event at the end of the file cannot corrupt
        # the reshape further down.
        if n_expected and len(event_rows) == n_expected:
            rows.setdefault(n_expected, []).extend(event_rows)
            states.setdefault(n_expected, []).append(tuple(event_names))

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("___EVENT_ID"):
                flush()
                n_expected = 0
                event_rows = []
                event_names = []
                continue

            if line.startswith("N_Particles"):
                n_expected = int(line.split(":")[1])
                continue

            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                values = [float(v) for v in parts[1:]]
            except ValueError:
                continue  # the "Particle Energy(TeV) ..." column header

            event_names.append(parts[0])
            event_rows.append(values)

    flush()

    return {n: Channel(n, rows[n], states[n]) for n in sorted(rows)}


# The name docs/report.pdf uses for each channel, for titles.
CHANNEL_NAMES = {2: "2 to 2", 3: "2 to 3", 4: "2 to 4"}


def channel_name(n_particles):
    return CHANNEL_NAMES.get(n_particles, f"2 to {n_particles}")


def iter_processes(channels):
    """Yield (state, Channel) once per distinct final state, in a fixed order.

    Ordered by particle count, then by how often the process occurs, so the
    output always comes out in the same sequence:

        p p -> p p,  p p -> gamma p p,  then the four 2 to 4 processes.
    """
    for n in sorted(channels):
        ch = channels[n]
        counts = Counter(ch.states)
        for state in sorted(counts, key=lambda s: (-counts[s], s)):
            yield state, ch.filter_state(state)


if __name__ == "__main__":
    # Quick look at what is in the file.
    channels = load_events()
    for n, ch in channels.items():
        print(f"{channel_name(n)}: {ch.n_events} events")
        for slot in range(n):
            names = ", ".join(ch.names_in_slot(slot))
            mean_E = ch.energy[:, slot].mean()
            print(f"    slot {slot} (particle {ch.particle_number(slot)}): "
                  f"mean E = {mean_E:7.4f} TeV   names = {names}")

    print()
    print("By process:")
    for state, sub in iter_processes(channels):
        print(f"    {process_ascii(state):<28}{sub.n_events:>7} events"
              f"   ->  plots/{process_slug(state)}/")
