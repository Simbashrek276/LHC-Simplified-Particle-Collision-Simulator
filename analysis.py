"""Apply detector cuts to a generated dataset.

simulation.py writes down every collision it produces, unfiltered. This file is
the second half of the job: it reads that dataset, throws away whatever the
detector would not have seen, and tells you what survived.

    simulation.py    generates events    -> collision_data/events.txt
    analysis.py     applies cuts        <- analysis_card.txt

Every cut value comes from analysis_card.txt, so changing what the detector sees
is a matter of editing that text file. Nothing in here needs touching.

Run:  python analysis.py
      python analysis.py my_other_card.txt      (to use a different card)
"""

import sys
from pathlib import Path

from analysis_tools import event_data
from utilities import kinematics

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CARD = PROJECT_ROOT / "analysis_card.txt"
# The cut events live with the analysis tools, separate from the raw events in
# collision_data/, so the folder name says which stage produced which file.
FILTERED_FILE = PROJECT_ROOT / "analysis_tools" / "analysis_data" / "events_analysed.txt"

# Every setting the card understands, with the value used if the card leaves it
# out. Anything not in here is rejected, so a typo becomes an error message
# rather than a cut that silently does nothing.
DEFAULTS = {
    "final_state": None,          # None keeps every collision type
    "min_energy": 0.0,
    "theta_min_deg": 0.0,
    "theta_max_deg": 180.0,
    "pseudo_rapidity_max": None,
    "transverse_momentum_min": 0.0,
    "write_filtered_file": True,
}

BOOLEAN_SETTINGS = {"write_filtered_file"}


# Section 1. Reading the card.

def read_card(path=DEFAULT_CARD):
    """Parse analysis_card.txt into a dictionary of cut values.

    The format is deliberately forgiving: `name = value`, `#` starts a comment,
    blank lines are skipped, and `none` switches a cut off. A misspelled setting
    name is an error rather than something we quietly ignore, because a cut that
    silently fails to apply is the worst possible outcome here.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Cannot find the analysis card at {path}\n"
            f"Expected a file called analysis_card.txt in the project root."
        )

    settings = dict(DEFAULTS)

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        if "=" not in line:
            raise SystemExit(
                f"{path.name} line {line_no}: expected 'name = value', got {raw.strip()!r}"
            )

        name, value = (part.strip() for part in line.split("=", 1))
        name = name.lower()

        if name not in DEFAULTS:
            known = ", ".join(sorted(DEFAULTS))
            raise SystemExit(
                f"{path.name} line {line_no}: unknown setting {name!r}.\n"
                f"Valid settings are: {known}"
            )

        settings[name] = _parse_value(name, value, path.name, line_no)

    _validate(settings)
    return settings


def _parse_value(name, value, filename, line_no):
    """Turn one card value into a number, a bool, or None."""
    low = value.lower()

    if name == "final_state":
        return _parse_final_state(value, filename, line_no)

    if name in BOOLEAN_SETTINGS:
        if low in ("yes", "true", "on", "1"):
            return True
        if low in ("no", "false", "off", "0"):
            return False
        raise SystemExit(
            f"{filename} line {line_no}: {name} should be yes or no, got {value!r}"
        )

    if low in ("none", "off", ""):
        # "off" means the cut is not applied. For a floor that is the same as
        # zero; for pseudorapidity there is no natural number, so we use None.
        return None if name == "pseudo_rapidity_max" else 0.0

    try:
        return float(value)
    except ValueError:
        raise SystemExit(
            f"{filename} line {line_no}: {name} should be a number or 'none', got {value!r}"
        )


def _parse_final_state(value, filename, line_no):
    """Turn the card's final_state value into a tuple of collision-type IDs.

    'all' (or 'none', or nothing) means keep every collision type, and comes back
    as None. Otherwise it is one ID or several separated by commas, such as
    '4' or '2, 6'. Each ID must be one of event_data.PROCESS_IDS.
    """
    if value.strip().lower() in ("all", "none", ""):
        return None

    valid = ", ".join(
        f"{pid} = {event_data.process_ascii(state)}"
        for pid, state in event_data.PROCESS_IDS.items()
    )
    ids = []
    for part in value.split(","):
        part = part.strip()
        try:
            pid = int(part)
        except ValueError:
            raise SystemExit(
                f"{filename} line {line_no}: final_state should be 'all' or collision-type "
                f"IDs separated by commas, got {value!r}.\nValid IDs: {valid}"
            )
        if pid not in event_data.PROCESS_IDS:
            raise SystemExit(
                f"{filename} line {line_no}: there is no collision type with ID {pid}.\n"
                f"Valid IDs: {valid}"
            )
        if pid not in ids:
            ids.append(pid)
    return tuple(sorted(ids))


def _validate(s):
    """Catch settings that are individually fine but cannot work together."""
    if s["theta_min_deg"] >= s["theta_max_deg"]:
        raise SystemExit(
            f"theta_min_deg ({s['theta_min_deg']}) must be below "
            f"theta_max_deg ({s['theta_max_deg']}), otherwise no particle can pass."
        )
    if not 0.0 <= s["theta_min_deg"] <= 180.0 or not 0.0 <= s["theta_max_deg"] <= 180.0:
        raise SystemExit("theta_min_deg and theta_max_deg must be between 0 and 180.")
    if s["min_energy"] < 0 or s["transverse_momentum_min"] < 0:
        raise SystemExit("min_energy and transverse_momentum_min cannot be negative.")
    if s["pseudo_rapidity_max"] is not None and s["pseudo_rapidity_max"] <= 0:
        raise SystemExit("pseudo_rapidity_max must be positive, or 'none' to switch off.")


def describe(settings):
    """A human readable summary of which cuts are actually switched on."""
    lines = []
    for pid in settings["final_state"] or ():
        state = event_data.PROCESS_IDS[pid]
        lines.append(f"  final state  = {pid}  ({event_data.process_ascii(state)})")
    if settings["min_energy"] > 0:
        lines.append(f"  energy      >= {settings['min_energy']} TeV")
    if settings["theta_min_deg"] > 0 or settings["theta_max_deg"] < 180:
        lines.append(
            f"  theta        in [{settings['theta_min_deg']}, "
            f"{settings['theta_max_deg']}] deg"
        )
    if settings["pseudo_rapidity_max"] is not None:
        lines.append(f"  |eta|        < {settings['pseudo_rapidity_max']}")
    if settings["transverse_momentum_min"] > 0:
        lines.append(f"  pt          >= {settings['transverse_momentum_min']} TeV")
    if not lines:
        lines.append("  (none -- every event passes)")
    return lines


# Section 2. Applying the cuts.

def particle_passes(energy, theta_deg, px, py, pz, settings):
    """True if one particle survives every cut that is switched on."""
    if energy < settings["min_energy"]:
        return False
    if not settings["theta_min_deg"] <= theta_deg <= settings["theta_max_deg"]:
        return False

    # pT and eta come from the physics engine, so the cuts use exactly the same
    # formulas as everything else in the project.
    p4 = kinematics.P4(energy, px, py, pz)

    if settings["transverse_momentum_min"] > 0:
        if kinematics.transverse_momentum(p4) < settings["transverse_momentum_min"]:
            return False

    if settings["pseudo_rapidity_max"] is not None:
        if abs(kinematics.pseudorapidity(p4)) >= settings["pseudo_rapidity_max"]:
            return False

    return True


def analyse(settings, data_file=None):
    """Apply the cuts to every channel. Returns per-channel and overall counts.

    An event is kept only if it is one of the collision types chosen with
    final_state, AND every one of its particles passes every cut.
    """
    channels = event_data.load_events(data_file or event_data.DEFAULT_FILE)

    # The final states the card asked for, or None to keep every type.
    chosen = None
    if settings["final_state"] is not None:
        chosen = {event_data.PROCESS_IDS[pid] for pid in settings["final_state"]}

    results = {}
    for n, ch in channels.items():
        kept_mask = []
        particles_kept = 0
        for i in range(ch.n_events):
            # Wrong collision type: drop the whole event without looking further.
            if chosen is not None and ch.states[i] not in chosen:
                kept_mask.append(False)
                continue

            flags = [
                particle_passes(
                    ch.energy[i, j], ch.theta[i, j],
                    ch.px[i, j], ch.py[i, j], ch.pz[i, j],
                    settings,
                )
                for j in range(n)
            ]
            particles_kept += sum(flags)
            kept_mask.append(all(flags))

        results[n] = {
            "channel": ch,
            "mask": kept_mask,
            "n_events": ch.n_events,
            "n_kept": sum(kept_mask),
            "n_particles": ch.n_events * n,
            "particles_kept": particles_kept,
        }
    return results


# Section 3. Reporting and writing.

def report(settings, results, card_path):
    print("=" * 68)
    print("ANALYSIS")
    print("=" * 68)
    print(f"Card:    {card_path}")
    print(f"Dataset: {event_data.DEFAULT_FILE}")
    print()
    print("Cuts applied:")
    for line in describe(settings):
        print(line)
    print("  event kept if: every particle must pass")
    print()

    header = f"{'channel':<12}{'events':>10}{'kept':>10}{'kept %':>10}{'particles kept %':>20}"
    print(header)
    print("-" * len(header))

    tot = tot_kept = tot_p = tot_pk = 0
    for n in sorted(results):
        r = results[n]
        pct = 100.0 * r["n_kept"] / r["n_events"] if r["n_events"] else 0.0
        ppct = 100.0 * r["particles_kept"] / r["n_particles"] if r["n_particles"] else 0.0
        print(f"{event_data.channel_name(n):<12}{r['n_events']:>10}{r['n_kept']:>10}"
              f"{pct:>9.1f}%{ppct:>19.1f}%")
        tot += r["n_events"]
        tot_kept += r["n_kept"]
        tot_p += r["n_particles"]
        tot_pk += r["particles_kept"]

    print("-" * len(header))
    pct = 100.0 * tot_kept / tot if tot else 0.0
    ppct = 100.0 * tot_pk / tot_p if tot_p else 0.0
    print(f"{'TOTAL':<12}{tot:>10}{tot_kept:>10}{pct:>9.1f}%{ppct:>19.1f}%")
    print()

    # With a final state chosen, the table above counts every event in the file,
    # so its percentages look small. This shows how the chosen types fared.
    if settings["final_state"] is not None:
        print("Chosen collision types:")
        for pid in settings["final_state"]:
            state = event_data.PROCESS_IDS[pid]
            in_file = kept = 0
            for r in results.values():
                for s, k in zip(r["channel"].states, r["mask"]):
                    if s == state:
                        in_file += 1
                        kept += k
            share = f"{100.0 * kept / in_file:.1f}%" if in_file else "-"
            print(f"  {pid}  {event_data.process_ascii(state):<26}"
                  f"{in_file:>7} in the file, {kept:>7} pass the cuts ({share})")
        print()
    return tot, tot_kept


def write_filtered(results):
    """Write the surviving events out in the same format simulation.py uses.

    Keeping the format identical means the graphing scripts can read this file
    without any changes -- point their DATA_FILE at it and everything works.
    """
    FILTERED_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(FILTERED_FILE, "w") as f:
        for n in sorted(results):
            r = results[n]
            ch = r["channel"]
            for i, keep in enumerate(r["mask"]):
                if not keep:
                    continue
                written += 1
                f.write(f"___EVENT_ID: {written}___\n")
                f.write(f"N_Particles: {n}\n")
                f.write(
                    f"{'Particle':<15}{'Energy(TeV)':<15}{'Theta(deg)':<15}"
                    f"{'Phi(deg)':<15}{'p_x(TeV)':<15}{'p_y(TeV)':<15}{'p_z(TeV)':<15}\n"
                )
                for j in range(n):
                    f.write(
                        f"{ch.states[i][j]:<15}{ch.energy[i, j]:<15.6f}"
                        f"{ch.theta[i, j]:<15.6f}{ch.phi[i, j]:<15.6f}"
                        f"{ch.px[i, j]:<15.6f}{ch.py[i, j]:<15.6f}{ch.pz[i, j]:<15.6f}\n"
                    )
                f.write("\n")
    return written


def main():
    card_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CARD
    settings = read_card(card_path)
    results = analyse(settings)
    total, kept = report(settings, results, card_path)

    if settings["write_filtered_file"]:
        written = write_filtered(results)
        print(f"Wrote {written} surviving events to:")
        print(f"  {FILTERED_FILE}")
        print()
        print("To plot these instead of the full sample, change DATA_FILE at the")
        print("top of the graph_*_steps.py scripts to point at")
        print("analysis_tools/analysis_data/events_analysed.txt.")
    else:
        print("write_filtered_file is off, so no file was written.")

    if kept == 0:
        print()
        print("WARNING: no events survived. Either the chosen final_state does not")
        print("occur in this dataset, or the cuts in the card are too tight.")


if __name__ == "__main__":
    main()
