# StarPlate

StarPlate is a local astronomical image workbench for examining detector pixels, checking stellar photometry and retaining image evidence. It ships two adjacent Sloan Digital Sky Survey r-band exposures so the worksheet can be opened without a network connection after installation.

## Run

Use Python 3.11 or 3.12. From the project directory:

```sh
python3 -m venv .venv
.venv/bin/pip install --only-binary=:all: -r requirements.txt
STARPLATE_PYTHON=.venv/bin/python ./start.sh
```

Open `http://127.0.0.1:19935/`. The start script accepts `PORT` and `STARPLATE_DATA`. To run the container instead, use `docker compose up --build`. The Compose service binds only to localhost and stores the frame library, marks and saved measurements in its named `work` volume. `docker compose stop` stops the service and preserves the volume. The optional `STARPLATE_PIP_INDEX` sets the wheel index for builds.

## Workbench

| Operation | Where to use it | Result |
| --- | --- | --- |
| Import and FITS plane selection | Import image, inspect HDUs | Select a supported 2D plane and add it to the library |
| Scientific display | Display | Preview the active frame using a linear or nonlinear stretch |
| Pixel probe | Point or exact original coordinates | Original value, local window and celestial coordinates when present |
| Celestial inspection | Sky grid, sky distance, footprint, WCS scale | World positions, separations, scale and detector support |
| Connected source finding | Find sources | Original location, source area, peak and integrated island intensity |
| Aperture photometry | Aperture flux, aperture growth curve | Fractionally weighted aperture flux, local sky estimate and uncertainty |
| Stellar shape | PSF moments, fit stellar profile, source shape map | FWHM, axes, elongation and fit diagnostics |
| Line and ribbon measurement | Line profile, strip statistics | Original values and valid support along the selected line |
| Intensity examination | Intensity histogram, compare two boxes | Distribution, quantiles and locations of representative bin pixels |
| Image quality | Detector quality, local contrast and noise tools | Region-local support, variation and detector anomalies |
| Isophotes and regions | Isolines, above-level area and region flux | Original-plane isolines or measurements within selected regions |
| Model and remove sky | Sky estimation and subtract sky | Cell evidence and a separate residual or sky image |
| Mask and filter | Mask unusable pixels, local median, band and edge filters | Separate processed image with the unaltered source still available |
| Camera calibration | Dark and flat correction, master calibration | Derived plane and reference coverage statistics |
| Align exposures | Align and combine, aligned residual check | Registration, overlap, residual evidence and a derived image |
| Seam mosaic | Seam mosaic | One admissible seam between two drafted endpoints, source-traced composite |
| Render color | RGB composite | Color preview and downloadable PNG using named input channels |
| Retain and export | Marks, saved measurements, frame exports | SQLite records, scientific FITS, PNG or evidence ZIP |

Coordinates are zero-based pixel centers in file row order. A box starts at its first included pixel center and its width and height count pixels. Preview stretch, screen size and color are display choices. Photometry and pixel readings use the stored original float32 values. Missing pixels are represented by NaN and exported with an optional VALID extension.

Supported inputs are two-dimensional numeric FITS image HDUs, including gzip and bzip2 files, and single-page PNG, TIFF and JPEG. RGB raster imports are converted to channel-weighted intensity and transparent pixels become unavailable. FITS data tables and image cubes are excluded from the import list. The maximum transfer is 46 MiB, decompressed FITS limit is 128 MiB, and image size is at most 8.5 million pixels with no edge above 8192. Individual tools have lower interactive region limits.

Stellar detection is a connected-island measurement and can group overlapping sources. Aperture error is an estimate from local scatter and optional gain. The sky tools are robust local or smooth mesh estimates and the shape tools assume simple profiles. Aperture accuracy depends on centering, valid support and surrounding sources. The WCS display is computed from celestial metadata when the FITS file supplies it. Resampling between exposures is an approximate pixel-level operation and does not perform flux-conserving area integration.

The seam mosaic builds on the same WCS (or shift/anchor) registration. You draft the join line by marking one endpoint on each vertical edge of the inspection box; the tool then chooses one seam row per column that connects the two endpoints, changes rows by at most one between adjacent columns, and stays within 18 pixels (configurable) above and below the drafted line. A seam position is admissible only when the seam row together with its immediate rows above and below carry complete finite samples below the brightness limit in both frames. Dynamic programming returns the single seam that best compromises between deviation from the drafted line and the original-value jump at the frame switch; a clear notice is returned when no admissible seam exists. In the retained composite the reference frame fills rows above the seam while the aligned companion fills the seam row and everything below, and any adopted pixel without a valid source sample is left empty. Reading input sources on the retained mosaic reports which frame supplied each pixel, its position on that original frame and the value actually read.

## Archive credit

The bundled archive files are unchanged calibrated SDSS DR17 corrected frames for rerun 301, run 2505, camera column 3, fields 38 and 39, in the r band. The original downloads are available from the Frame tab. Credit goes to the Sloan Digital Sky Survey. The current SDSS image use policy describes public data release data as public domain.

- [SDSS DR17 imaging data](https://www.sdss4.org/dr17/imaging/images/)
- [SDSS image use policy](https://sdss.org/collaboration/image-use-policy/)
- [SDSS archive field 38](https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-r-002505-3-0038.fits.bz2)
- [SDSS archive field 39](https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-r-002505-3-0039.fits.bz2)
