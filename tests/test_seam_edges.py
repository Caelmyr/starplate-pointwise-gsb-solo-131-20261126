import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401  (path / astropy-stub bootstrap)
import tempfile
import numpy as np
from astropy.io import fits
from starplate.web import create_app
from starplate.frames import ImagePlane

rng = np.random.default_rng(1)
H, W = 30, 40
a = rng.normal(0, .1, (H,W)).astype(np.float32)
b = (a + .1).astype(np.float32)
tmp = tempfile.mkdtemp()
app = create_app({"TESTING": True, "STARPLATE_DATA": tmp, "STARPLATE_ARCHIVE": tempfile.mkdtemp(), "SEED_ARCHIVE": False})
st = app.config["FRAME_STORE"]
h = fits.Header()
st.store(ImagePlane(a, h.copy(), "fits", 0, "c", ()), title="a", original_name="a.fits", origin="upload", digest="e"*64, key="ka", lineage={})
st.store(ImagePlane(b, h.copy(), "fits", 0, "c", ()), title="b", original_name="b.fits", origin="upload", digest="f"*64, key="kb", lineage={})
c = app.test_client()
base = {"box": {"x":0,"y":0,"width":W,"height":H}, "partner":"kb", "alignment":"shift",
        "shift_x":0,"shift_y":0,"interpolation":"nearest","support":.999,"blend":"reference",
        "alpha":.5,"normalize":"none","require_pair":False,"brightness_limit":1e6,"band":18}
# reversed endpoint order (right first)
r = c.post("/api/frames/ka/derive/seam", json={**base, "start":[W-1, 12], "end":[0, 10]})
assert r.status_code == 200, r.get_data(as_text=True)
seam = r.get_json()["result"]["seam"]
assert seam[0] == [0,10] and seam[-1] == [W-1,12]
print("reversed endpoints handled")
# sub-pixel endpoints within tolerance
r = c.post("/api/frames/ka/derive/seam", json={**base, "start":[0.2, 10.4], "end":[W-1-0.2, 12.6]})
assert r.status_code == 200, r.get_data(as_text=True)
seam = r.get_json()["result"]["seam"]
assert seam[0] == [0,10] and seam[-1] == [W-1,13]
print("sub-pixel snapping:", seam[0], seam[-1])
# tolerance exceeded
r = c.post("/api/frames/ka/derive/seam", json={**base, "start":[1.0, 10], "end":[W-1,12]})
assert r.status_code == 400 and "left edge" in r.get_json()["error"]
print("outside tolerance rejected")
# seam in a 2-row-tall box: no interior rows -> infeasible
r = c.post("/api/frames/ka/derive/seam", json={**base,
    "box": {"x":0,"y":10,"width":W,"height":2}, "start":[0,10], "end":[W-1,11]})
assert r.status_code == 400
print("2-row box rejected:", r.get_json()["error"][:60])
# different units rejected
h2 = fits.Header()
st.store(ImagePlane(b, h2.copy(), "fits", 0, "jy", ()), title="b2", original_name="b2.fits",
         origin="upload", digest="9"*64, key="kc", lineage={})
r = c.post("/api/frames/ka/derive/seam", json={**base, "partner":"kc", "start":[0,10],"end":[W-1,12]})
assert r.status_code == 400 and "units" in r.get_json()["error"]
print("unit mismatch rejected")
# same frame rejected
r = c.post("/api/frames/ka/derive/seam", json={**base, "partner":"ka", "start":[0,10],"end":[W-1,12]})
assert r.status_code == 400
print("same-frame rejected")
print("EDGE TESTS PASSED")
