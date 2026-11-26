import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401  (path / astropy-stub bootstrap)
import numpy as np
from starplate.registration import Alignment, Blend
from starplate.seam import SeamSettings, find_seam, _candidate_bounds
from starplate.errors import RequestError

# Endpoint near top row: interior clipping must not crash
rng = np.random.default_rng(0)
H,W=8,12
ref=rng.normal(0,.1,(H,W)).astype(np.float32); aln=(ref+.1).astype(np.float32)
sup=np.ones((H,W),np.float32)
align=Alignment("shift",0,0,(),()); blend=Blend("reference","nearest",.999,.5,"none",False)
st=SeamSettings((0,1),(W-1,1),1e6,align,blend,18)
seam,guide,_,_=find_seam(ref,aln,sup,st,1,1)
assert seam[0]==1 and seam[-1]==1 and np.all(seam>=1)
print("near-top endpoint OK:", seam.tolist())
# near bottom
st=SeamSettings((0,H-2),(W-1,H-2),1e6,align,blend,18)
seam,_,_,_=find_seam(ref,aln,sup,st,H-2,H-2)
assert np.all(seam<=H-2)
print("near-bottom endpoint OK")
# wide band larger than frame
st=SeamSettings((0,3),(W-1,4),1e6,align,blend,64)
seam,_,_,_=find_seam(ref,aln,sup,st,3,4)
assert seam[0]==3 and seam[-1]==4
print("oversized band OK")
# empty candidate column simulation: bounds can invert
lo,hi=_candidate_bounds(5, 10, 8, 1, 6, 2)
print("bounds sample:", lo, hi)
# No overlap at all -> NaN aligned everywhere
aln_nan=np.full((H,W),np.nan,np.float32); sup0=np.zeros((H,W),np.float32)
try:
    find_seam(ref,aln_nan,sup0,SeamSettings((0,3),(W-1,4),1e6,align,blend,18),3,4)
    raise AssertionError("should fail")
except RequestError as e:
    print("all-NaN partner rejected:", str(e)[:60])
# endpoint itself NaN in reference
ref2=ref.copy(); ref2[0:3,0]=np.nan
try:
    find_seam(ref2,aln,sup,SeamSettings((0,1),(W-1,1),1e6,align,blend,18),1,1)
    raise AssertionError("should fail")
except RequestError as e:
    print("bad endpoint rejected:", str(e)[:60])
print("EDGE2 PASSED")

# Inverted bounds (no reachable rows in a column) must yield clean infeasibility, not crashes
rng2 = np.random.default_rng(2)
H2,W2=10,14
r2=rng2.normal(0,.1,(H2,W2)).astype(np.float32); a2=(r2+.1).astype(np.float32)
# block column 7 completely (all rows saturated in both frames at the triple condition)
r2[:,6:9]=99.0; a2[:,6:9]=99.0
sup2=np.ones((H2,W2),np.float32)
st2=SeamSettings((0,5),(W2-1,5),50.0,align,blend,18)
try:
    find_seam(r2,a2,sup2,st2,5,5)
    raise AssertionError("expected infeasible")
except RequestError as e:
    print("full-width block infeasible:", str(e)[:70])
print("INVERTED BOUNDS PASSED")
