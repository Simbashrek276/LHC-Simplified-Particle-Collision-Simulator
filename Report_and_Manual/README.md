# Report_and_Manual/

This folder holds the project's written report, **`report.pdf`**. It serves as
both the report and the manual: it contains the full physics derivations behind
the simulation, and explains how the program works.

The code and the main README refer to it by section and equation number — for
example `utilities/kinematics.py` is built on section 2.5, and its Lorentz boost
is the 3D generalisation of equation 8. If a link to
`Report_and_Manual/report.pdf` is broken, the report has not been added to this
copy of the repository yet.

The `figures/` subfolder holds the images used in the report's "Example of
Numerical Results" section. They are drawn with the project's own graph scripts
from a separate run of 1,000,000 events (seed 2026), using only the events that
pass all of the default cuts. That run is too large to keep on GitHub; to
regenerate it, set `TARGET_LOGGED_EVENTS = 1000000` in `simulation.py` and point
`OUTPUT_FILE` at `collision_data/events_1M.txt` (already in `.gitignore`).
