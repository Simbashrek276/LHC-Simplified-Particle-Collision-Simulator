"""Apply detector cuts to a generated dataset.

collision.py writes down every collision it produces, unfiltered. This file is
the second half of the job: it reads that dataset, throws away whatever the
detector would not have seen, and tells you what survived.

    collision.py    generates events    -> collision_data/events.txt
    analysis.py     applies cuts        <- analysis_card.txt

Every cut value comes from analysis_card.txt, so changing what the detector sees
is a matter of editing that text file. Nothing in here needs touching.

Run:  python analysis.py
      python analysis.py my_other_card.txt      (to use a different card)
"""

import math
import sys
from pathlib import Path

from analysis_tools import event_data

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CARD = PROJECT_ROOT / "analysis_card.txt"
FILTERED_FILE = PROJECT_ROOT / "collision_data" / "events_analysed.txt"

# Every setting the card understands, with the value used if the card leaves it
# out. Anything not in here is rejected, so a typo becomes an error message
# rather than a cut that silently does nothing.
DEFAULTS = {
    "min_energy": 0.0,
    "theta_min_deg": 0.0,
    "theta_max_deg": 180.0,
    "pseudo_rapidity_max": None,
    "transverse_momentum_min": 0.0,
    "require_all_particles": True,
    "write_filtered_file": True,
}

BOOLEAN_SETTINGS = {"require_all_particles", "write_filtered_file"}


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
    if settings["min_energy"] > 0:
        lines.append(f"  energy      >= {settings['min_energy']} TeV")
    if settings["theta_min_deg"] > 0 or settings["theta_max_deg"] < 180:
        lines.append(
            f"  theta        in [{settings['theta_min_deg']}, "
            f"{settings['theta_max_deg']}] deg"
        )
    if settings["pseudo_rapidity_max"] is not None:
        eta = settings["pseudo_rapidity_max"]
        lo = math.degrees(2 * math.atan(math.exp(-eta)))
        lines.append(f"  |eta|        < {eta}   (theta in [{lo:.3f}, {180 - lo:.3f}] deg)")
    if settings["transverse_momentum_min"] > 0:
        lines.append(f"  pt          >= {settings['transverse_momentum_min']} TeV")
    if not lines:
        lines.append("  (none -- every particle passes)")
    return lines


# Section 2. Applying the cuts.

def particle_passes(energy, theta_deg, px, py, pz, settings):
    """True if one particle survives every cut that is switched on."""
    if energy < settings["min_energy"]:
        return False
    if not settings["theta_min_deg"] <= theta_deg <= settings["theta_max_deg"]:
        return False

    if settings["transverse_momentum_min"] > 0:
        pt = math.sqrt(px * px + py * py)
        if pt < settings["transverse_momentum_min"]:
            return False

    if settings["pseudo_rapidity_max"] is not None:
        # eta = 0.5*ln((|p|+pz)/(|p|-pz)), which equals -ln(tan(theta/2)) but
        # does not blow up when theta is tiny. A particle exactly on the beam
        # line has infinite eta and correctly fails any finite cut.
        p_mag = math.sqrt(px * px + py * py + pz * pz)
        if p_mag <= 0 or abs(pz) >= p_mag:
            return False
        eta = 0.5 * math.log((p_mag + pz) / (p_mag - pz))
        if abs(eta) >= settings["pseudo_rapidity_max"]:
            return False

    return True


def analyse(settings, data_file=None):
    """Apply the cuts to every channel. Returns per-channel and overall counts."""
    channels = event_data.load_events(data_file or event_data.DEFAULT_FILE)
    strict = settings["require_all_particles"]

    results = {}
    for n, ch in channels.items():
        kept_mask = []
        particles_kept = 0
        for i in range(ch.n_events):
            flags = [
                particle_passes(
                    ch.energy[i, j], ch.theta[i, j],
                    ch.px[i, j], ch.py[i, j], ch.pz[i, j],
                    settings,
                )
                for j in range(n)
            ]
            particles_kept += sum(flags)
            kept_mask.append(all(flags) if strict else any(flags))

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
    mode = "every particle must pass" if settings["require_all_particles"] \
        else "at least one particle must pass"
    print(f"  event kept if: {mode}")
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
    return tot, tot_kept


def write_filtered(results):
    """Write the surviving events out in the same format collision.py uses.

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
        print("top of the graph_*_steps.py scripts to point at events_analysed.txt.")
    else:
        print("write_filtered_file is off, so no file was written.")

    if kept == 0:
        print()
        print("WARNING: no events survived. The cuts in the card are too tight.")


if __name__ == "__main__":
    main()
