import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401  (path / astropy-stub bootstrap)
import tempfile
import numpy as np
from astropy.io import fits
from starplate.web import create_app
from starplate.frames import ImagePlane

rng = np.random.default_rng(7)
H, W = 50, 70
base = 3 + .01*np.indices((H,W))[0]
ref = (base + rng.normal(0,.1,(H,W))).astype(np.float32)
# partner shifted by (1.5, -0.5) relative to reference
from scipy.ndimage import shift as ndshift
part = ndshift(base + .2 + rng.normal(0,.1,(H,W)), (-.5, 1.5), order=1, cval=np.nan).astype(np.float32)

tmp = tempfile.mkdtemp()
app = create_app({"TESTING": True, "STARPLATE_DATA": tmp, "STARPLATE_ARCHIVE": tempfile.mkdtemp(),
                  "SEED_ARCHIVE": False})
store = app.config["FRAME_STORE"]
hdr = fits.Header()
store.store(ImagePlane(ref, hdr.copy(), "fits", 0, "count", ()), title="ref",
            original_name="r.fits", origin="upload", digest="c"*64, key="ref-x", lineage={})
store.store(ImagePlane(part, hdr.copy(), "fits", 0, "count", ()), title="part",
            original_name="p.fits", origin="upload", digest="d"*64, key="part-x", lineage={})
client = app.test_client()

# Inspection box not at origin
box = {"x": 8, "y": 6, "width": 50, "height": 36}
payload = {
    "box": box, "partner": "part-x", "alignment": "shift",
    "shift_x": 1.5, "shift_y": -0.5, "interpolation": "linear", "support": .999,
    "blend": "reference", "alpha": .5, "normalize": "none", "require_pair": False,
    "start": [8, 20], "end": [57, 22], "brightness_limit": 1e6, "band": 18,
}
r = client.post("/api/frames/ref-x/derive/seam", json=payload)
assert r.status_code == 200, r.get_data(as_text=True)
result = r.get_json()["result"]
seam = result["seam"]
assert seam[0][0] == 8 and seam[-1][0] == 57
# retain
r = client.post("/api/frames/ref-x/derive/seam", json={**payload, "retain": True, "title": "M"})
assert r.status_code in (200, 201)
mkey = r.get_json()["frame"]["key"]
mosaic = store.open(mkey)
mp = mosaic.pixels
assert mp.shape == (36, 50)
lineage = mosaic.record.lineage
seam_local = lineage["parameters"]["seam_rows"]
assert lineage["origin"] == [8, 6]
assert lineage["region_on_parent"] == box

# Verify composite numerically against direct resampling for a few columns
from starplate.registration import Alignment, Blend, transform_for, transformed_patch
frame_r = store.open("ref-x"); frame_p = store.open("part-x")
from starplate.geometry import Box
b = Box(8, 6, 50, 36)
align = Alignment("shift", 1.5, -.5, (), ())
blend = Blend("reference", "linear", .999, .5, "none", False)
mapping, _ = transform_for(frame_r, frame_p, b, align)
aligned, support = transformed_patch(frame_p, b, mapping, blend)
ref_plane = b.extract(frame_r.pixels)
for lc in range(50):
    sr = seam_local[lc]
    for lr in range(36):
        expected = aligned[lr, lc] if lr >= sr else ref_plane[lr, lc]
        if np.isnan(expected):
            assert np.isnan(mp[lr, lc]), (lr, lc)
        else:
            assert abs(float(mp[lr, lc]) - float(expected)) < 1e-6, (lr, lc)
print("linear-interpolation composite matches resampled inputs exactly")

# Provenance at a partner-side point with sub-pixel mapping
gx, gy = 8 + 20, 6 + 30  # global point; local (20,30)
r = client.post(f"/api/frames/{mkey}/source-samples", json={"point": [20, 30]})
trace = r.get_json()
src = trace["adopted_source"]
# local row 30 vs seam at column 20
uses_partner = 30 >= seam_local[20]
assert (src["role"] == "partner") == uses_partner
if uses_partner:
    assert src["available"] == np.isfinite(aligned[30, 20])
    assert abs(src["point"][0] - (gx + 1.5)) < 1e-6 and abs(src["point"][1] - (gy - .5)) < 1e-6
    if src["available"]:
        assert abs(src["value"] - float(aligned[30, 20])) < 1e-6
        assert src["interpolation_support"] > 0
    print("sub-pixel partner trace:", src)
# reference-side point
r = client.post(f"/api/frames/{mkey}/source-samples", json={"point": [10, 2]})
trace = r.get_json()
assert trace["adopted_source"]["role"] == "reference"
assert trace["adopted_source"]["value"] == float(frame_r.pixels[6+2, 8+10])
print("reference trace with offset origin OK")

# history manifest still works
r = client.get(f"/api/frames/{mkey}/history")
assert r.status_code == 200
manifest = r.get_json()
assert manifest["frames"][0]["record"]["lineage"]["operation"] == "seam"
print("INTEGRATION 2 PASSED")
