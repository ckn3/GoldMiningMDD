#!/deac/opt/rocky9-noarch/anaconda3/bin/python
"""
Plot GoldMDD site polygons on a light OpenStreetMap basemap.

Reads the spatial metadata table from GoldMDD/README.md so the plot stays aligned
with current GoldMDD naming (including PlayaMirador1/2 and Paolita).
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.patches import Patch, Rectangle
from PIL import Image


ROOT = Path("/deac/csc/alqahtaniGrp/cuij")
GOLDMDD = ROOT / "GoldMDD"
README = GOLDMDD / "README.md"
OUT_PNG = GOLDMDD / "site_distribution_osm.png"

TILE_SIZE = 256
MAX_LAT = 85.05112878
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "GoldMDD-Map/1.0"


@dataclass
class SitePoly:
    name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


def _clamp_lat(lat: float) -> float:
    return max(-MAX_LAT, min(MAX_LAT, lat))


def lonlat_to_world_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = _clamp_lat(lat)
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * (n * TILE_SIZE)
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (n * TILE_SIZE)
    return x, y


def lonlat_to_tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    x, y = lonlat_to_world_pixel(lon, lat, zoom)
    return int(math.floor(x / TILE_SIZE)), int(math.floor(y / TILE_SIZE))


def tile_xy_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def bbox_tile_range(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int) -> tuple[int, int, int, int]:
    tx0, ty1 = lonlat_to_tile_xy(min_lon, min_lat, zoom)
    tx1, ty0 = lonlat_to_tile_xy(max_lon, max_lat, zoom)
    return min(tx0, tx1), min(ty0, ty1), max(tx0, tx1), max(ty0, ty1)


def tile_count_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int) -> int:
    tx_min, ty_min, tx_max, ty_max = bbox_tile_range(min_lon, min_lat, max_lon, max_lat, zoom)
    return (tx_max - tx_min + 1) * (ty_max - ty_min + 1)


def choose_zoom(min_lon: float, min_lat: float, max_lon: float, max_lat: float, max_tiles: int = 120) -> int:
    for z in range(16, 4, -1):
        if tile_count_for_bbox(min_lon, min_lat, max_lon, max_lat, z) <= max_tiles:
            return z
    return 5


def fetch_tile(tx: int, ty: int, zoom: int, session: requests.Session) -> np.ndarray:
    url = OSM_URL.format(z=zoom, x=tx, y=ty)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return np.asarray(Image.open(BytesIO(resp.content)).convert("RGB"))


def build_osm_mosaic(min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    tx_min, ty_min, tx_max, ty_max = bbox_tile_range(min_lon, min_lat, max_lon, max_lat, zoom)
    canvas = np.full(((ty_max - ty_min + 1) * TILE_SIZE, (tx_max - tx_min + 1) * TILE_SIZE, 3), 255, dtype=np.uint8)
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            y0 = (ty - ty_min) * TILE_SIZE
            x0 = (tx - tx_min) * TILE_SIZE
            try:
                canvas[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = fetch_tile(tx, ty, zoom, sess)
            except Exception:
                pass
    ul_lon, ul_lat = tile_xy_to_lonlat(tx_min, ty_min, zoom)
    lr_lon, lr_lat = tile_xy_to_lonlat(tx_max + 1, ty_max + 1, zoom)
    # Lighten background so filled polygons remain readable.
    canvas = (0.58 * canvas.astype(np.float32) + 0.42 * 255.0).clip(0, 255).astype(np.uint8)
    return canvas, (ul_lon, lr_lon, lr_lat, ul_lat)


def _parse_range(cell: str) -> tuple[float, float]:
    vals = ast.literal_eval(cell.strip())
    if not isinstance(vals, (list, tuple)) or len(vals) != 2:
        raise ValueError(f"Bad lon/lat range cell: {cell}")
    return float(vals[0]), float(vals[1])


def parse_spatial_table_from_readme(readme_path: Path) -> list[SitePoly]:
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "## Spatial metadata table (source files + aligned output)":
            start = i
            break
    if start is None:
        raise RuntimeError("Spatial metadata heading not found in README")

    # Find table header and rows.
    i = start + 1
    while i < len(lines) and not lines[i].startswith("| Site |"):
        i += 1
    if i >= len(lines):
        raise RuntimeError("Spatial metadata table header not found")
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 2  # skip separator
    rows: list[dict[str, str]] = []
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
        i += 1

    explicit_playa_split = any(str(row.get("Site", "")).strip() in {"PlayaMirador1", "PlayaMirador2"} for row in rows)

    out: list[SitePoly] = []
    for row in rows:
        site = row["Site"]
        lon0, lon1 = _parse_range(row["Label lon range"])
        lat0, lat1 = _parse_range(row["Label lat range"])
        min_lon, max_lon = min(lon0, lon1), max(lon0, lon1)
        min_lat, max_lat = min(lat0, lat1), max(lat0, lat1)

        if site == "PlayaMirador" and not explicit_playa_split:
            mid_lat = 0.5 * (min_lat + max_lat)
            # top half = higher latitude, bottom half = lower latitude
            out.append(SitePoly("PlayaMirador1", min_lon, min_lat, max_lon, mid_lat))
            out.append(SitePoly("PlayaMirador2", min_lon, mid_lat, max_lon, max_lat))
        else:
            out.append(SitePoly(site, min_lon, min_lat, max_lon, max_lat))
    return out


def plot_map(polys: list[SitePoly], out_png: Path) -> None:
    min_lon = min(p.min_lon for p in polys)
    min_lat = min(p.min_lat for p in polys)
    max_lon = max(p.max_lon for p in polys)
    max_lat = max(p.max_lat for p in polys)
    pad_x = (max_lon - min_lon) * 0.08
    pad_y = (max_lat - min_lat) * 0.08
    min_lon -= pad_x
    max_lon += pad_x
    min_lat -= pad_y
    max_lat += pad_y

    z = choose_zoom(min_lon, min_lat, max_lon, max_lat, max_tiles=120)
    img, extent = build_osm_mosaic(min_lon, min_lat, max_lon, max_lat, z)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=180)
    ax.imshow(img, extent=extent, origin="upper")

    cmap = plt.get_cmap("tab20")
    legend_patches: list[Patch] = []
    for i, p in enumerate(polys):
        c = cmap(i % 20)
        fill = (c[0], c[1], c[2], 0.40)
        rect = Rectangle(
            (p.min_lon, p.min_lat),
            p.max_lon - p.min_lon,
            p.max_lat - p.min_lat,
            linewidth=0.0,
            edgecolor="none",
            facecolor=fill,
        )
        ax.add_patch(rect)
        legend_patches.append(Patch(facecolor=(c[0], c[1], c[2], 0.55), edgecolor="none", label=p.name))

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    polys = parse_spatial_table_from_readme(README)
    # Stable order matching README rows, with PlayaMirador split inserted.
    plot_map(polys, OUT_PNG)
    print(f"Wrote: {OUT_PNG}")


if __name__ == "__main__":
    main()
