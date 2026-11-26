import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401  (path / astropy-stub bootstrap)
import numpy as np
from starplate.registration import Alignment, Blend
from starplate.seam import (
    SeamSettings, find_seam, _eligibility, _candidate_bounds, _guide_for, _jump_scale,
)
from starplate.errors import RequestError

def brute(ref_p, aln_p, elig, left, right, band, jscale):
    h, w = aln_p.shape
    guide = _guide_for(left, right, w)
    best = [np.inf, None]
    def rec(col, prev, path, cost):
        if col == w:
            if cost < best[0]: best[0], best[1] = cost, path[:]
            return
        lo, hi = _candidate_bounds(col, w, h, left, right, band)
        for r in range(lo, hi+1):
            if col == 0:
                if r != left: continue
            elif abs(r-prev) > 1: continue
            if not elig[r, col]: continue
            node = abs(r-guide[col])/band
            jump = 0.0 if col == 0 else abs(float(ref_p[r,col])-float(aln_p[r,col]))/jscale
            rec(col+1, r, path+[r], cost+node+jump)
    rec(0, None, [], 0.0)
    return best[1], best[0]

rng = np.random.default_rng(123)
trials = 1500
mismatch = 0
infeasible = 0
for t in range(trials):
    h = int(rng.integers(6, 13)); w = int(rng.integers(5, 12))
    base = rng.normal(0, 1, (h, w))
    ref = (base + rng.normal(0, rng.uniform(.02,.5), (h,w))).astype(np.float32)
    aln = (base + rng.uniform(-1,1) + rng.normal(0, rng.uniform(.02,.5), (h,w))).astype(np.float32)
    # randomly punch some saturated/invalid cells
    for _ in range(int(rng.integers(0, 6))):
        cy, cx = int(rng.integers(1,h-1)), int(rng.integers(0,w))
        if rng.random() < .5:
            ref[cy, cx] = 99
        else:
            aln[cy, cx] = np.nan
    band = int(rng.integers(2, 6)); limit = float(rng.uniform(5, 50))
    align = Alignment("wcs",0,0,(),()); blend = Blend("reference","linear",.999,.5,"none",False)
    left = int(rng.integers(2, h-2)); right = int(np.clip(left + rng.integers(-2,3), 2, h-3))
    st = SeamSettings((0,left),(w-1,right),limit,align,blend,band)
    support = np.where(np.isnan(aln), 0.0, 1.0).astype(np.float32)
    elig,_,_ = _eligibility(ref, aln, support, st)
    jscale = _jump_scale(ref.astype(float), aln.astype(float), elig)
    try:
        seam, guide, _, _ = find_seam(ref, aln, support, st, left, right)
    except RequestError:
        infeasible += 1
        bf, _ = brute(ref, aln, elig, left, right, band, jscale)
        assert bf is None, f"trial {t}: DP said infeasible but brute found {bf}"
        continue
    # structural checks
    assert seam[0] == left and seam[-1] == right
    assert np.all(np.abs(np.diff(seam)) <= 1)
    band_dev = np.max(np.abs(seam - guide))
    assert band_dev <= band + 1e-9, f"trial {t}: band deviation {band_dev} > {band}"
    for c in range(w):
        for r in (seam[c]-1, seam[c], seam[c]+1):
            assert np.isfinite(ref[r,c]) and ref[r,c] < limit, f"trial {t} ref violation r{r} c{c}"
            assert np.isfinite(aln[r,c]) and aln[r,c] < limit, f"trial {t} aln violation r{r} c{c}"
    # optimality
    bf, bcost = brute(ref, aln, elig, left, right, band, jscale)
    g = _guide_for(left,right,w)
    dpcost = sum(abs(seam[c]-g[c])/band + (0 if c==0 else abs(float(ref[seam[c],c])-float(aln[seam[c],c]))/jscale) for c in range(w))
    if abs(dpcost - bcost) > 1e-9:
        mismatch += 1
        print(f"trial {t} MISMATCH dp={dpcost} brute={bcost}\n dp={seam.tolist()}\n bf={bf}")
        if mismatch > 3: break
print(f"trials={trials} infeasible-consistent={infeasible} cost-mismatches={mismatch}")
assert mismatch == 0
print("FUZZ PASSED")
