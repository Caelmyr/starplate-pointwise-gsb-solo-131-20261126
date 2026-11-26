"""Test bootstrap for environments without the full runtime on sys.path.

In the normal project virtual environment (see requirements.txt) astropy,
Flask, numpy and scipy are installed and no path setup is needed.  In an
offline sandbox those packages may live in a separate target directory and
astropy may be unavailable, in which case the tiny compatibility stub under
``tests/_astropy_stub`` is used so the seam logic can still be exercised.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_SITE_PACKAGES = (
    Path("/opt/pointwise-env/site-packages"),
)
STUB = Path(__file__).resolve().parent / "_astropy_stub"

for candidate in CANDIDATE_SITE_PACKAGES:
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import astropy  # noqa: F401
except ModuleNotFoundError:
    if STUB.is_dir() and str(STUB) not in sys.path:
        sys.path.insert(0, str(STUB))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
