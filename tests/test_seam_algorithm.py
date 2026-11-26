import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401  (path / astropy-stub bootstrap)

import numpy as np
from starplate.geometry import Box
from starplate.registration import Alignment, Blend
from starplate.seam import (
    SeamSettings, find_seam, compose, _eligibility, _ordered_endpoints,
    _candidate_bounds, _jump_scale, BAND_RADIUS,
)

rng = np.random.default_rng(42)
H, W = 60, 80

def make_settings(start, end, limit, band=18, interp="linear"):
    align = Alignment("wcs", 0, 0, (), ())
    blend = Blend("reference", interp, 0.999, 0.5, "none", False)
    return SeamSettings(start, end, limit, align, blend, band)

# Smooth background shared by both frames + independent noise + offset between frames.
base = 10 + 0.01*np.indices((H, W))[1]
ref = (base + rng.normal(0, 0.05, (H, W))).astype(np.float32)
partner = (base + 0.3 + rng.normal(0, 0.05, (H, W))).astype(np.float32)

# Bright star near the middle of the drafted seam in BOTH frames.
for cy in (28, 29):
    for cx in (38, 39, 40):
        ref[cy, cx] = 50
        partner[cy, cx] = 55

box = Box(0, 0, W, H)
settings = make_settings((0, 30), (W-1, 32), limit=20.0)

ref_plane = box.extract(ref)
aligned = partner.copy()
support = np.ones((H, W), dtype=np.float32)

seam, guide, jscale, dscale = find_seam(ref_plane, aligned, support, settings, 30, 32)
print("seam first/last:", seam[0], seam[-1])
assert seam[0] == 30 and seam[-1] == 32
steps = np.diff(seam)
assert np.all(np.abs(steps) <= 1), "slope must be at most one row per column"
deviation = np.abs(seam - np.rint(guide))
assert np.max(deviation) <= 18, "seam left the band"
# Seam must route around the bright star: no seam row within the star block rows 27..30 at cols 37..41
for c in range(37, 42):
    triple_ok = all(ref[r, c] < 20 and aligned[r, c] < 20 for r in (seam[c]-1, seam[c], seam[c]+1))
    assert triple_ok, f"seam column {c} violates the brightness/sample rule"
print("max deviation from guide:", deviation.max(), "mean:", deviation.mean())
print("seam rows around star (cols 35-44):", seam[35:45])

# Eligibility mask sanity
elig, fref, fpart = _eligibility(ref_plane, aligned, support, settings)
assert not elig[0].any() and not elig[H-1].any(), "border rows can never be eligible"
assert elig[10, 10]

# Composite rules
composite, nref, npart = compose(ref_plane, aligned, seam)
for c in range(W):
    for r in range(H):
        if r < seam[c]:
            assert composite[r, c] == ref[r, c]
        else:
            assert composite[r, c] == aligned[r, c]
assert np.isfinite(composite).all()
print("composite reference/partner counts:", nref, npart)

# Missing partner samples -> NaN in partner region
aligned2 = aligned.copy()
aligned2[40:45, 10:15] = np.nan
composite2, _, _ = compose(ref_plane, aligned2, seam)
assert np.isnan(composite2[42, 12]) and np.isfinite(ref_plane[42, 12])
assert np.isfinite(composite2[10, 12])  # reference region untouched
print("missing-sample blanking ok")

# Endpoint validation
try:
    _ordered_endpoints(make_settings((5, 30), (W-1, 32), 20.0), box)
    raise AssertionError("expected edge validation error")
except Exception as e:
    assert "left edge" in str(e)
try:
    _ordered_endpoints(make_settings((0, 30), (W-2, 32), 20.0), box)
    raise AssertionError("expected edge validation error")
except Exception as e:
    assert "right edge" in str(e)
print("endpoint validation ok")

# No feasible seam: block every candidate path at one column with saturated rows
blocked = partner.copy()
blocked[:, 50] = 99
blocked[:, 49] = 99
blocked[:, 51] = 99
try:
    find_seam(ref_plane, blocked, support, settings, 30, 32)
    raise AssertionError("expected no-seam error")
except Exception as e:
    print("no-seam message:", str(e)[:80])

# Endpoint itself blocked
blocked2 = partner.copy()
blocked2[29:32, 0] = 99
try:
    find_seam(ref_plane, blocked2, support, settings, 30, 32)
    raise AssertionError("expected endpoint error")
except Exception as e:
    print("endpoint message:", str(e)[:80])

# Tight band forces near-guide path
tight = make_settings((0, 30), (W-1, 30), 20.0, band=2)
seam_t, guide_t, _, _ = find_seam(ref_plane, aligned, support, tight, 30, 30)
assert np.max(np.abs(seam_t - np.rint(guide_t))) <= 2
print("tight band ok, rows 35-44:", seam_t[35:45])

# Pure cost behavior: identical frames -> jump zero everywhere, minimum deviation path = guide
same = ref.copy()
s3, g3, _, _ = find_seam(ref_plane, same.astype(np.float32), support,
                         make_settings((0, 20), (W-1, 40), 1e9), 20, 40)
assert np.max(np.abs(s3 - g3)) <= 0.6, f'deviation {np.max(np.abs(s3-g3))}'
print('identical-frame max deviation:', np.max(np.abs(s3-g3)))
print("guide-following on identical frames ok")
print("ALL ALGORITHM TESTS PASSED")

# Brute-force optimality check on a small grid against exhaustive path enumeration.
import itertools
def brute_force(ref_p, aln_p, settings, left, right):
    h, w = aln_p.shape
    elig, _, _ = _eligibility(ref_p, aln_p, np.ones_like(aln_p, dtype=np.float32), settings)
    guide = left + (right-left)*np.arange(w)/max(1,w-1)
    jscale = _jump_scale(ref_p.astype(np.float64), aln_p.astype(np.float64), elig)
    band = settings.band
    best_cost, best_path = np.inf, None
    def recurse(col, prev_row, path, cost):
        nonlocal best_cost, best_path
        if col == w:
            if cost < best_cost:
                best_cost, best_path = cost, path[:]
            return
        low, high = _candidate_bounds(col, w, h, left, right, band)
        for r in range(low, high+1):
            if col == 0:
                if r != left:
                    continue
            elif abs(r - prev_row) > 1:
                continue
            if not elig[r, col]:
                continue
            node = abs(r - guide[col]) / band
            jump = 0.0 if col == 0 else abs(float(ref_p[r, col]) - float(aln_p[r, col])) / jscale
            recurse(col+1, r, path+[r], cost+node+jump)
    recurse(0, None, [], 0.0)
    return best_path, best_cost

rng2 = np.random.default_rng(7)
h2, w2 = 9, 7
base2 = rng2.normal(0, 1, (h2, w2))
ref2 = (base2 + rng2.normal(0, 0.3, (h2, w2))).astype(np.float32)
aln2 = (base2 + 0.4 + rng2.normal(0, 0.3, (h2, w2))).astype(np.float32)
st2 = make_settings((0, 4), (w2-1, 5), 1e6, band=3)
seam2, guide2, js2, ds2 = find_seam(ref2, aln2, np.ones((h2,w2), np.float32), st2, 4, 5)
bf_path, bf_cost = brute_force(ref2, aln2, st2, 4, 5)
print("DP  :", seam2.tolist())
print("brute:", bf_path)
def path_cost(path):
    g = 4 + (5-4)*np.arange(w2)/max(1,w2-1)
    total = abs(path[0]-g[0])/st2.band
    for col in range(1, w2):
        r = path[col]
        total += abs(r-g[col])/st2.band + abs(float(ref2[r,col])-float(aln2[r,col]))/js2
    return total
dp_cost = path_cost(seam2.tolist())
print("DP cost:", dp_cost, "brute cost:", bf_cost)
assert abs(dp_cost - bf_cost) < 1e-9, (dp_cost, bf_cost)
assert seam2.tolist() == bf_path or abs(dp_cost - bf_cost) < 1e-9
print("DP matches brute-force optimum (cost equal; ties allowed)")

# Brute force with a blocked region and tighter band
aln2b = aln2.copy()
aln2b[3:6, 3] = 99
ref2b = ref2.copy()
ref2b[3:6, 3] = 99
st2b = make_settings((0, 4), (w2-1, 5), 50.0, band=4)
seam2b, _, _, _ = find_seam(ref2b, aln2b, np.ones((h2,w2), np.float32), st2b, 4, 5)
bf2, _ = brute_force(ref2b, aln2b, st2b, 4, 5)
assert seam2b.tolist() == bf2
for r in (seam2b[3]-1, seam2b[3], seam2b[3]+1):
    assert ref2b[r,3] < 50 and aln2b[r,3] < 50
print("blocked-column DP matches brute force:", seam2b.tolist())
print("ALL BRUTE-FORCE TESTS PASSED")
