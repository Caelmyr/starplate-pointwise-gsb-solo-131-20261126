import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401  (path / astropy-stub bootstrap)
import tempfile

import numpy as np
from astropy.io import fits  # stub

from starplate.web import create_app
from starplate.frames import ImagePlane
from starplate.store import FrameStore

rng = np.random.default_rng(99)
H, W = 48, 64
base = 5 + 0.02*np.indices((H,W))[1]
ref_pixels = (base + rng.normal(0, .08, (H,W))).astype(np.float32)
part_pixels = (base + .25 + rng.normal(0, .08, (H,W))).astype(np.float32)
# bright star near the drafted seam -> must be routed around, and above limit
ref_pixels[22:25, 30:33] = 80
part_pixels[22:25, 30:33] = 85
# missing partner samples in lower region -> composite blanks there
part_pixels[40:43, 5:9] = np.nan

tmp = tempfile.mkdtemp()
app = create_app({"TESTING": True, "STARPLATE_DATA": tmp, "STARPLATE_ARCHIVE": tempfile.mkdtemp(),
                  "SEED_ARCHIVE": False})
store = app.config["FRAME_STORE"]
hdr = fits.Header()
ref_plane = ImagePlane(ref_pixels, hdr.copy(), "fits", 0, "count", ())
part_plane = ImagePlane(part_pixels, hdr.copy(), "fits", 0, "count", ())
ref_rec, _ = store.store(ref_plane, title="reference", original_name="ref.fits",
                         origin="upload", digest="a"*64, key="ref-1", lineage={})
part_rec, _ = store.store(part_plane, title="partner", original_name="part.fits",
                          origin="upload", digest="b"*64, key="part-1", lineage={})

client = app.test_client()

# 1. Preview seam via derive endpoint (no retain)
payload = {
    "box": {"x": 0, "y": 0, "width": W, "height": H},
    "partner": "part-1", "alignment": "shift", "shift_x": 0, "shift_y": 0, "interpolation": "nearest",
    "support": .999, "blend": "reference", "alpha": .5, "normalize": "none",
    "require_pair": False,
    "start": [0, 24], "end": [W-1, 26],
    "brightness_limit": 30.0, "band": 18,
}
r = client.post("/api/frames/ref-1/derive/seam", json=payload)
assert r.status_code == 200, r.get_data(as_text=True)
data = r.get_json()["result"]
seam = data["seam"]
assert len(seam) == W
assert seam[0] == [0, 24] and seam[-1] == [W-1, 26]
rows = np.array([y for x, y in seam])
assert np.all(np.abs(np.diff(rows)) <= 1)
assert np.max(np.abs(rows - (24 + (26-24)*np.arange(W)/(W-1)))) <= 18
# star avoided: every seam triple below limit in both frames
for c, (x, y) in enumerate(seam):
    for rr in (y-1, y, y+1):
        assert np.isfinite(ref_pixels[rr, x]) and ref_pixels[rr, x] < 30
        assert np.isfinite(part_pixels[rr, x]) and part_pixels[rr, x] < 30
print("preview seam OK; max deviation", data["deviation"]["max_px"], "mean jump", data["jump"]["mean"])

# 2. Retain the composite
payload_retain = dict(payload, retain=True, title="Seam mosaic")
r = client.post("/api/frames/ref-1/derive/seam", json=payload_retain)
assert r.status_code in (200, 201), r.get_data(as_text=True)
created = r.get_json()
mosaic_key = created["frame"]["key"]
print("retained mosaic:", mosaic_key)

# 3. Open the mosaic and verify pixel contents directly
mosaic = store.open(mosaic_key)
mp = mosaic.pixels
lineage = mosaic.record.lineage
assert lineage["operation"] == "seam"
assert lineage["parents"] == ["ref-1", "part-1"]
seam_local = lineage["parameters"]["seam_rows"]
assert len(seam_local) == W
for c in range(W):
    sr = seam_local[c]
    for rr in range(H):
        if rr < sr:
            assert mp[rr, c] == ref_pixels[rr, c] or (np.isnan(mp[rr,c]) and np.isnan(ref_pixels[rr,c]))
        else:
            if np.isnan(part_pixels[rr, c]):
                assert np.isnan(mp[rr, c]), f"blank expected at {rr},{c}"
            else:
                assert mp[rr, c] == part_pixels[rr, c]
# explicit blank check
assert np.isnan(mp[41, 7]) and np.isfinite(ref_pixels[41, 7])
print("composite pixel rules and blanking verified")

# 4. Provenance trace above the seam -> reference
r = client.post(f"/api/frames/{mosaic_key}/source-samples", json={"point": [10, 5]})
assert r.status_code == 200, r.get_data(as_text=True)
trace = r.get_json()
assert trace["operation"] == "seam"
assert trace["adopted_source"]["role"] == "reference"
assert trace["adopted_source"]["key"] == "ref-1"
assert trace["adopted_source"]["value"] == float(ref_pixels[5, 10])
print("above-seam trace -> reference:", trace["adopted_source"]["value"])

# 5. Provenance trace below the seam -> partner with original position
r = client.post(f"/api/frames/{mosaic_key}/source-samples", json={"point": [12, 40]})
assert r.status_code == 200, r.get_data(as_text=True)
trace = r.get_json()
src = trace["adopted_source"]
assert src["role"] == "partner" and src["key"] == "part-1"
# nearest-neighbour identity mapping in stub WCS means partner position ~ same coords
assert src["available"] and src["value"] == float(part_pixels[40, 12])
assert src["pixel"] == [12, 40]
print("below-seam trace -> partner:", src)

# 6. Trace at the blanked partner pixel -> not available, empty
r = client.post(f"/api/frames/{mosaic_key}/source-samples", json={"point": [7, 41]})
assert r.status_code == 200, r.get_data(as_text=True)
trace = r.get_json()
assert trace["adopted_source"]["role"] == "partner"
assert trace["adopted_source"]["available"] is False
assert trace["result"]["value"] is None
print("blank-pixel trace reports unavailable adopted source")

# 7. Trace exactly on the seam row -> partner (seam row belongs to companion)
seam_y = seam_local[20]
r = client.post(f"/api/frames/{mosaic_key}/source-samples", json={"point": [20, seam_y]})
trace = r.get_json()
assert trace["adopted_source"]["role"] == "partner"
assert trace["adopted_source"]["value"] == float(part_pixels[seam_y, 20])
print("on-seam-row trace -> partner OK")

# 8. No feasible seam: impossible brightness limit
bad = dict(payload, brightness_limit=-1e9)
r = client.post("/api/frames/ref-1/derive/seam", json=bad)
assert r.status_code == 400
print("no-seam correctly rejected:", r.get_json()["error"][:70])

# 9. Endpoint off the box edges -> validation error
bad2 = dict(payload, start=[3, 24])
r = client.post("/api/frames/ref-1/derive/seam", json=bad2)
assert r.status_code == 400 and "left edge" in r.get_json()["error"]
print("edge validation OK")

# 10. Probe in preview report
probe_payload = dict(payload, point=[12, 40])
r = client.post("/api/frames/ref-1/derive/seam", json=probe_payload)
report = r.get_json()["result"]
assert report["probe"]["source"] == "partner"
assert report["probe"]["partner_value"] == float(part_pixels[40, 12])
print("preview probe OK:", report["probe"])

print("INTEGRATION PASSED")
