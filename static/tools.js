const n = (key, label, value, step = "any") => ({key, label, value, step, type: "number"});
const str = (key, label, value = "") => ({key, label, value, type: "text"});
const select = (key, label, options, value = options[0]) => ({key, label, options, value, type: "select"});
const check = (key, label, value = false) => ({key, label, value, type: "checkbox"});
const image = (key, label, optional = false) => ({key, label, optional, type: "frame"});
const list = (key, label, value = "") => ({key, label, value, type: "list"});
const point = (key, label, value = "") => ({key, label, value, type: "point"});
const referenceList = {key: "references", label: "Other exposures", type: "frames"};

const aperture = [
  n("radius", "Aperture radius (px)", 4.5), n("sky_inner", "Sky inner radius", 7.5),
  n("sky_outer", "Sky outer radius", 12.5), n("clip_sigma", "Sky clipping σ", 3),
  n("min_aperture_coverage", "Minimum aperture coverage", .90), n("min_sky_coverage", "Minimum sky coverage", .65),
  n("gain", "Gain (0 if unknown)", 0),
];

const star = [n("threshold", "Detection threshold σ", 4), n("smoothing", "Smoothing σ (px)", .8),
  n("min_area", "Min pixels", 4, 1), n("max_area", "Max pixels", 350, 1), n("limit", "Max sources", 150, 1)];
const line = [n("spacing", "Samples every (px)", 1), select("interpolation", "Sampling", ["linear", "nearest"])];
const sky = [select("method", "Sky model", ["mesh", "constant", "plane"]), n("tile_size", "Cell size (px)", 64, 1),
  n("clip_sigma", "Clipping σ", 3), n("min_coverage", "Minimum cell coverage", .55), n("smooth", "Mesh smoothness σ", .65)];
const levels = [
  select("output", "Output plane", [["residual", "Residual"], ["background", "Background surface"]]),
];
const quality = [n("cell", "Cell size (px)", 96, 1), n("min_support", "Minimum support", .7), n("clip", "Clipping σ", 3),
  n("saturation", "Saturation (optional)", "")];
const shape = [select("shape", "Region shape", ["ellipse", "box", "polygon"]), n("major", "Ellipse semi-major", 18),
  n("minor", "Ellipse semi-minor", 12), n("angle", "Ellipse angle (°)", 0)];
const alignment = [image("partner", "Companion image"),
  select("alignment", "Alignment", ["wcs", "shift", "phase", "anchors"]),
  n("shift_x", "Shift X", 0), n("shift_y", "Shift Y", 0),
  point("anchor_1", "Image anchor 1 (x, y)"), point("anchor_2", "Image anchor 2 (x, y)"),
  point("moving_1", "Companion anchor 1"), point("moving_2", "Companion anchor 2"),
  select("interpolation", "Interpolation", ["linear", "nearest"]), n("support", "Minimum source support", .999)];
const channels = ["red", "green", "blue"].flatMap((name) => [
  image(`${name}.frame`, `${name} image`, true), n(`${name}.weight`, `${name} weight`, 1),
  n(`${name}.low_pct`, `${name} low percentile`, 5), n(`${name}.high_pct`, `${name} high percentile`, 99.7),
]);
const extraBox = ["x", "y", "width", "height"].map((name) => n(`reference.${name}`, `Reference ${name}`, name.length === 1 ? 0 : 30, 1));

export const groups = {
  "Photometry": {
    "aperture": ["Aperture flux", "Read a source with a weighted circular aperture and a clipped local sky annulus.", [...aperture, check("recenter", "Recenter around the click")], "point"],
    "stars": ["Find sources", "Measure connected sources inside the inspection box. Select a row to read or mark its original location.", star],
    "aperture-curve": ["Aperture growth curve", "Compare flux as the same aperture grows, holding its sky annulus fixed.", [...aperture, n("curve_start", "First radius (px)", 1), n("samples", "Samples", 12, 1)], "point"],
    "psf": ["PSF moments", "Read the core shape around the active point using original-image moments.", [n("radius", "Core radius (px)", 8), n("clipping", "Core threshold σ", 2.5)], "point"],
    "fit-source": ["Fit stellar profile", "Fit an elliptical Gaussian to a small original-pixel neighborhood.", [
      n("radius", "Fit radius (px)", 8, 1), n("max_offset", "Maximum center shift", 3),
      select("background", "Background", ["constant", "plane"]), check("fix_center", "Hold center fixed"), check("positive", "Fit positive peak", true),
    ], "point"],
    "source-shape-map": ["Source shape map", "Summarize focus and common elongation direction across detected sources.", [...star, n("minimum_minor", "Minimum minor FWHM", 1.7), n("elongated", "Elongation threshold", .16)]],
    "region-flux": ["Region flux", "Read the sum and source extrema within an ellipse, user polygon or rectangular box.", [...shape, n("sky", "Known sky level", 0)]],
    "region-centroids": ["Above-level centroids", "Locate above-level connected regions inside a specified shape.", [...shape, n("level", "Original level", .1), n("limit", "Max sources", 20, 1)]],
    "rings": ["Three radial bands", "Summarize original samples inside three concentric, disjoint bands.", [n("first", "First edge (px)", 5), n("middle", "Middle edge (px)", 11),
      n("last", "Outer edge (px)", 20), n("sky", "Known sky level", 0)], "point"],
  },
  "Pixels and coordinates": {
    "probe": ["Pixel probe", "Read original intensity, nearby pixels and celestial coordinates where available.", [n("radius", "Neighborhood radius", 1, 1), n("patch", "Grid half-size", 4, 1)], "point"],
    "contamination-band": ["Satellite contamination band · 卫星污染带", "Use Line to join the two still-separate faint-trail peaks, then find evidenced peaks and left/right original shoulders on each detector row. Crossing rows with doubtful shoulders are left undecided.", [
      n("drop_electrons", "Minimum peak-to-shoulder drop (e−)", 40, 1),
      n("row_search_radius", "Peak search radius (px)", 10, 1),
      n("max_half_width", "Maximum shoulder half-width (px)", 18, 1),
      n("trace_tolerance", "Trace tolerance (px)", 3, .1),
      n("min_sky_samples", "Clean sky samples per side", 3, 1),
    ], "line"],
    "profile": ["Line profile", "Plot original brightness along a line. A wider ribbon averages around its axis.", [...line, n("width", "Ribbon width (odd px)", 1, 1)], "line"],
    "strip": ["Strip statistics", "Compare mean and median brightness along and across an offset ribbon.", [...line, n("half_width", "Strip half-width", 7, 1)], "line"],
    "sky-distance": ["Sky distance", "Measure detector and celestial separation between line endpoints.", [], "line"],
    "world": ["Find sky coordinate", "Locate an ICRS right ascension and declination on the frame.", [str("ra", "Right ascension"), str("dec", "Declination")]],
    "wcs-scale": ["WCS scale grid", "Check local plate scale and direction over the image.", [n("columns", "Grid columns", 5, 1), n("rows", "Grid rows", 4, 1)]],
    "footprint": ["Sky footprint", "Read the celestial perimeter and angular area of the inspection box.", [n("samples", "Samples per edge", 12, 1)]],
    "observation": ["Observation metadata", "Read timestamp, detector metadata and exposure settings available in the FITS header.", []],
  },
  "Image inspection": {
    "histogram": ["Intensity histogram", "See the original intensity distribution and select a bin to find representative detector pixels.", [
      n("bins", "Bin count", 96, 1), n("low", "Lower limit (auto)", ""), n("high", "Upper limit (auto)", ""),
    ]],
    "compare": ["Compare two boxes", "Compare intensity and noise between the current box and a second region of this frame.", extraBox],
    "quality": ["Detector quality map", "Map coverage, local sky and adjacent-pixel noise across the inspection box.", quality],
    "detector-noise": ["Row and column noise", "Read per-row or per-column statistics and report prominent level steps.", [quality[1], select("axis", "Direction", ["row", "column"]), n("seam_sigma", "Flag step above σ", 2.7)]],
    "contrast-cells": ["Local contrast grid", "Compare the actual percentile spread of each detector cell.", [n("cell", "Cell size (px)", 80, 1), n("low_pct", "Low percentile", 5), n("high_pct", "High percentile", 95)]],
    "defects": ["Hot and cold pixels", "Locate small anomalous regions relative to their local median.", [n("sigma", "Cutoff σ", 6),
      n("neighborhood", "Local median size (odd px)", 7, 1), n("max_area", "Maximum area", 16, 1),
      n("limit", "Marks to display", 65, 1), select("direction", "Anomaly direction", ["both", "hot", "cold"])]],
    "extrema": ["Local extrema", "Identify pixels with the strongest or weakest values in their neighborhoods.", [select("sign", "Direction", ["highest", "lowest"]), n("level", "Cutoff", .12), n("isolation", "Isolation (px)", 4, 1), n("limit", "Max points", 40, 1)]],
    "contours": ["Isolines", "Trace original-image crossings at user-entered intensity levels. Missing cells break contours.", [
      list("levels", "Levels, separated by commas", ".1,.15,.2"), n("minimum_length", "Minimum trace length", 7),
    ]],
    "isophote": ["Above-level area", "Summarize the area and flux of connected bright regions at one intensity.", [n("level", "Intensity level", .15), n("min_area", "Minimum area", 6, 1)]],
    "background": ["Sky estimation", "Inspect the sampled background surface and its residuals. Click a point to include a local readout.", sky],
  },
  "Derived exposures": {
    "sky": ["Subtract sky", "Compute a sky-subtracted or sky-only image from the inspection box.", [...sky, ...levels], "", "derive"],
    "mask": ["Mask unusable pixels", "Mark original pixels as missing. Retaining the preview creates a separate detector plane.", [
      select("mode", "Mask rule", [["outside_range", "Outside intensity range"], ["inside_range", "Inside intensity range"], ["polygon", "Inside polygon"]]),
      n("lower", "Original low", -.05), n("upper", "Original high", .15), n("dilate", "Grow mask (px)", 0, 1),
    ], "", "derive"],
    "detail": ["Local median detail", "Use a median surface to isolate local structure in the box.", [n("window", "Odd median window (px)", 9, 1), ...levels], "", "derive"],
    "multiscale": ["Band and edge filters", "Estimate original-image structure at chosen spatial scales.", [
      select("filter", "Filter", ["bandpass", "gaussian", "laplacian", "gradient"]), n("fine", "Fine scale (px)", 1.1),
      n("broad", "Broad scale (px)", 4.5), n("minimum_mass", "Minimum filter support", .9), check("signed", "Keep signed response", true),
    ], "", "derive"],
    "crop": ["Keep a cutout", "Copy the inspection box with scientific intensities and shifted WCS.", [], "", "derive"],
    "calibrate": ["Dark and flat correction", "Apply matching camera reference images. The box supplies the flat normalization sample.", [
      image("dark", "Dark frame", true), image("flat", "Flat frame", true), n("dark_scale", "Dark scale", 1),
      n("flat_min_fraction", "Min flat response", .25), n("flat_max_fraction", "Max flat response", 2), n("minimum_reference", "Min normalization pixels", 20, 1),
    ], "", "derive"],
    "master": ["Master calibration", "Combine calibration exposures in fixed detector coordinates.", [
      referenceList, select("mode", "Reference type", [["median", "Median dark"], ["normalized_flat", "Normalized flat"]]),
    ], "", "derive"],
    "register": ["Align and combine", "Map the companion image into the original detector grid, then combine their overlapping samples.", [
      ...alignment, select("blend", "Result", ["mean", "difference", "maximum", ["reference", "Aligned companion"]]),
      select("normalize", "Match levels", ["none", "offset", "affine"]), n("alpha", "Companion weight", .5), check("require_pair", "Only keep paired pixels"),
    ], "", "derive"],
    "pair": ["Aligned residual check", "Test the agreement and original support of an aligned pair before combining.", [
      ...alignment, n("cell", "Cell size", 64, 1), n("min_joint", "Min paired samples", 30, 1),
      check("level_match", "Match background offset", true),
    ], "", "direct"],
    "pair-footprint": ["Companion footprint", "Map the original detector border onto the companion exposure.", [...alignment, n("samples", "Points per edge", 40, 1)], "", "direct"],
    "color": ["RGB composite", "Combine three exposures as color layers with independent original-intensity ranges. Blank layer choices use the active frame.", [
      select("mode", "Stretch mode", ["linked", "per-channel"]), select("alignment", "Alignment", ["detector", "wcs"]),
      n("stretch", "Stretch strength", 5), n("saturation", "Color saturation", 1), n("luminance", "Luminance", 1), ...channels,
    ], "", "color"],
  },
};

export const tools = Object.assign({}, ...Object.values(groups));
