from dataclasses import dataclass
import logging

from .frames import read_plane, source_digest


@dataclass(frozen=True)
class ArchiveEntry:
    key: str
    name: str
    title: str
    url: str


SDSS = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3"
ARCHIVE = (
    ArchiveEntry(
        "sdss-r-0038",
        "sdss-002505-3-0038-r.fits.bz2",
        "SDSS r · 2505/3/38",
        f"{SDSS}/frame-r-002505-3-0038.fits.bz2",
    ),
    ArchiveEntry(
        "sdss-r-0039",
        "sdss-002505-3-0039-r.fits.bz2",
        "SDSS r · 2505/3/39",
        f"{SDSS}/frame-r-002505-3-0039.fits.bz2",
    ),
)


def initialize_archive(store, sample_dir):
    for entry in ARCHIVE:
        path = sample_dir / entry.name
        if not path.is_file():
            logging.getLogger(__name__).warning("Bundled archive frame missing: %s", entry.name)
            continue
        raw = path.read_bytes()
        digest = source_digest(raw)
        if store.existing(digest, 0, "archive"):
            continue
        plane = read_plane(entry.name, raw, 0)
        store.store(
            plane,
            key=entry.key,
            title=entry.title,
            original_name=entry.name,
            origin="archive",
            digest=digest,
            source_url=entry.url,
            lineage={"archive": "SDSS DR17", "parents": []},
        )
