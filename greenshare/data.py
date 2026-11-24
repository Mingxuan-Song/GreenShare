"""Download and join official Chicago park, population, and boundary data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import get_coordinates, make_valid, transform, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape

DATASETS = {
    "population": {
        "id": "t68z-cikk", "filename": "population.json", "format": "json",
        "query": {"$where": "acs_year=2023", "$order": "community_area", "$limit": 5000},
        "count_query": {"$where": "acs_year=2023", "$select": "count(*) as n"},
    },
    "communities": {
        "id": "igwz-8jzy", "filename": "communities.geojson", "format": "geojson",
        "query": {"$order": "area_numbe", "$limit": 5000},
        "count_query": {"$select": "count(*) as n"},
    },
    "parks": {
        "id": "ejsh-fztr", "filename": "parks.geojson", "format": "geojson",
        "query": {"$order": "objectid_1", "$limit": 5000},
        "count_query": {"$select": "count(*) as n"},
    },
}
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
PROJECTED_CRS = "EPSG:26916"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _fetch_json(url: str, path: Path) -> dict:
    """Use verified HTTPS, bounded reads, and retries; reject HTML error pages."""
    last_error = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "GreenShareResearch/1.0", "Accept": "application/json"})
            with urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                content = response.read(MAX_DOWNLOAD_BYTES + 1)
                if len(content) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Response exceeds the 20 MiB per-file limit")
                parsed = json.loads(content)
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("The response must be a JSON object or array")
                record = {
                    "url": url, "final_url": response.url, "http_status": response.status,
                    "content_type": response.headers.get("Content-Type", ""),
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                    "checked_utc": datetime.now(timezone.utc).isoformat(),
                    "filename": path.name,
                }
            path.write_bytes(content)
            return record
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download and parse {url}: {last_error}") from last_error


def download_all(root: Path, refresh: bool = False) -> dict:
    """Get all three complete sources, or verify their existing cached hashes.

    An existing manifest is an offline cache contract. A changed or missing file
    raises an error instead of silently downloading different data.
    """
    root = Path(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            path = raw / record["filename"]
            if not path.is_file() or _sha256(path) != record["sha256"]:
                raise ValueError(f"Cached data missing or changed: {path.name}; use --refresh to redownload")
        validate_sources(raw)
        return manifest

    staging = Path(tempfile.mkdtemp(prefix="download-", dir=raw))
    manifest = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "population_year": 2023,
        "park_inventory_date": "2016-11-04",
        "files": [], "datasets": {},
    }
    try:
        for name, spec in DATASETS.items():
            source_id = spec["id"]
            metadata_url = f"https://data.cityofchicago.org/api/views/{source_id}.json"
            data_url = f"https://data.cityofchicago.org/resource/{source_id}.{spec['format']}?{urlencode(spec['query'])}"
            count_url = f"https://data.cityofchicago.org/resource/{source_id}.json?{urlencode(spec['count_query'])}"
            paths = [staging / f"{name}_metadata.json", staging / f"{name}_count.json", staging / spec["filename"]]
            for url, path in zip([metadata_url, count_url, data_url], paths):
                manifest["files"].append(_fetch_json(url, path))
            metadata = json.loads(paths[0].read_text())
            count_rows = json.loads(paths[1].read_text())
            expected = int(count_rows[0]["n"])
            content = json.loads(paths[2].read_text())
            actual = len(content if isinstance(content, list) else content["features"])
            if actual != expected or actual >= int(spec["query"]["$limit"]):
                raise ValueError(f"{name}: possible truncation, {actual} rows downloaded vs {expected} reported")
            if metadata.get("id") != source_id:
                raise ValueError(f"Incorrect metadata source for {name}")
            manifest["datasets"][name] = {
                "source_id": source_id, "landing_page": f"https://data.cityofchicago.org/d/{source_id}",
                "title": metadata.get("name"), "description": metadata.get("description"),
                "license_id": metadata.get("licenseId"), "attribution": metadata.get("attribution"),
                "terms_url": "https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html",
                "downloaded_rows": actual, "reported_rows": expected,
                "rows_updated_at": metadata.get("rowsUpdatedAt"),
            }
        manifest["validation"] = validate_sources(staging)
        for record in manifest["files"]:
            os.replace(staging / record["filename"], raw / record["filename"])
        _write_json(manifest_path, manifest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def _normalize_name(value: object) -> str:
    return " ".join(str(value).upper().replace("'", "").replace("’", "").split())


def _load_features(path: Path, required_properties: set[str]) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError(f"{path.name}: expected a GeoJSON FeatureCollection")
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path.name}: no features")
    for feature in features:
        if feature.get("type") != "Feature" or not required_properties <= set(feature.get("properties", {})):
            raise ValueError(f"{path.name}: missing feature properties")
        geometry = feature.get("geometry", {})
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"{path.name}: non-polygon geometry")
        polygon = shape(geometry)
        xy = get_coordinates(polygon)
        if polygon.is_empty or not np.isfinite(xy).all():
            raise ValueError(f"{path.name}: empty or non-finite geometry")
        if np.any(np.abs(xy[:, 0]) > 180) or np.any(np.abs(xy[:, 1]) > 90):
            raise ValueError(f"{path.name}: coordinates are outside longitude/latitude ranges")
        xmin, ymin, xmax, ymax = polygon.bounds
        if not (-89 < xmin < xmax < -86 and 41 < ymin < ymax < 43):
            raise ValueError(f"{path.name}: geometry falls outside the Chicago study envelope")
    return features


def validate_sources(raw: Path) -> dict:
    """Validate schemas, finite populations, identifiers, and one-to-one join."""
    raw = Path(raw)
    population = json.loads((raw / "population.json").read_text(encoding="utf-8"))
    if not isinstance(population, list) or len(population) != 77:
        raise ValueError("Expected 77 population rows for ACS 2023")
    names = []
    for row in population:
        if not {"acs_year", "community_area", "total_population", "record_id"} <= set(row):
            raise ValueError("Missing population columns")
        if float(row["acs_year"]) != 2023:
            raise ValueError("Unexpected population year")
        count = float(row["total_population"])
        if not math.isfinite(count) or count <= 0:
            raise ValueError("Population must be finite and strictly positive")
        names.append(_normalize_name(row["community_area"]))
    if len(set(names)) != 77 or len({r["record_id"] for r in population}) != 77:
        raise ValueError("Duplicate population names or record IDs")
    communities = _load_features(raw / "communities.geojson", {"community", "area_numbe"})
    parks = _load_features(raw / "parks.geojson", {"objectid_1", "park_no", "park"})
    community_names = [_normalize_name(f["properties"]["community"]) for f in communities]
    ids = [int(f["properties"]["area_numbe"]) for f in communities]
    if len(communities) != 77 or set(ids) != set(range(1, 78)) or len(set(community_names)) != 77:
        raise ValueError("Expected unique community area IDs 1 through 77 and names")
    if set(names) != set(community_names):
        raise ValueError("Population and boundary community names do not match exactly")
    if len({f["properties"]["objectid_1"] for f in parks}) != len(parks):
        raise ValueError("Duplicate park object IDs")
    return {"population_rows": len(population), "community_features": len(communities),
            "park_features": len(parks), "joined_communities": 77,
            "population_total": sum(float(row["total_population"]) for row in population)}


def _polygon_only(geometry):
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [_polygon_only(g) for g in geometry.geoms]
        return union_all([p for p in polygons if p is not None and not p.is_empty])
    return None


def prepare_data(root: Path) -> tuple[pd.DataFrame, list, dict]:
    """Return area table, EPSG:26916 geometries, and measured quality checks."""
    root = Path(root)
    download_all(root)
    raw = root / "data" / "raw"
    quality = validate_sources(raw)
    population = json.loads((raw / "population.json").read_text())
    pop = {_normalize_name(row["community_area"]): float(row["total_population"]) for row in population}
    communities = _load_features(raw / "communities.geojson", {"community", "area_numbe"})
    communities.sort(key=lambda f: int(f["properties"]["area_numbe"]))
    parks = _load_features(raw / "parks.geojson", {"objectid_1", "park_no", "park"})
    project = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform
    unproject = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True).transform
    repaired = {"communities": 0, "parks": 0}

    def convert(features, kind):
        output = []
        for feature in features:
            geom = shape(feature["geometry"])
            if not geom.is_valid:
                geom = _polygon_only(make_valid(geom))
                repaired[kind] += 1
            if geom is None or geom.is_empty:
                raise ValueError(f"Repair left an empty {kind} geometry")
            projected = transform(geom, project, interleaved=False)
            if not projected.is_valid:
                projected = _polygon_only(make_valid(projected))
            if projected is None or projected.is_empty or projected.area <= 0:
                raise ValueError(f"Invalid projected {kind} geometry")
            output.append(projected)
        return output

    area_geometries = convert(communities, "communities")
    park_geometries = convert(parks, "parks")
    city_union = union_all(area_geometries)
    park_union = union_all(park_geometries)
    clipped_park_union = park_union.intersection(city_union)
    records = []
    for feature, geom in zip(communities, area_geometries):
        properties = feature["properties"]
        name = _normalize_name(properties["community"])
        park_area = float(geom.intersection(park_union).area)
        area = float(geom.area)
        if park_area < 0 or park_area > area * (1 + 1e-9):
            raise ValueError("Park intersection exceeds a community boundary")
        records.append({
            "area_id": int(properties["area_numbe"]), "name": name,
            "population": pop[name], "area_m2": area, "park_m2": park_area,
            "park_per_person": park_area / pop[name], "park_fraction": park_area / area,
            "pop_density_km2": pop[name] / area * 1e6,
        })
    table = pd.DataFrame(records)
    overlap_area = sum(g.area for g in area_geometries) - city_union.area
    park_overlap = sum(g.area for g in park_geometries) - park_union.area
    intersection_components = []
    for geom in area_geometries:
        intersection = _polygon_only(geom.intersection(park_union))
        if intersection is not None and not intersection.is_empty:
            intersection_components.extend(intersection.geoms if isinstance(intersection, MultiPolygon) else [intersection])
    quality.update({
        "population_year": 2023, "park_inventory_date": "2016-11-04",
        "projected_crs": PROJECTED_CRS, "invalid_geometries_repaired": repaired,
        "park_raw_sum_m2": float(sum(g.area for g in park_geometries)),
        "park_union_m2": float(park_union.area),
        "park_overlapping_area_removed_m2": max(0.0, float(park_overlap)),
        "park_clipped_union_m2": float(clipped_park_union.area),
        "park_outside_communities_m2": float(park_union.area - clipped_park_union.area),
        "park_sum_across_communities_m2": float(table.park_m2.sum()),
        "community_sum_m2": float(table.area_m2.sum()),
        "community_union_m2": float(city_union.area),
        "community_overlap_m2": max(0.0, float(overlap_area)),
        "park_assignment_overlap_m2": max(0.0, float(table.park_m2.sum() - clipped_park_union.area)),
        "intersection_fragments_under_1m2": int(sum(g.area < 1 for g in intersection_components)),
        "fragment_area_under_1m2": float(sum(g.area for g in intersection_components if g.area < 1)),
        "sliver_policy": "All polygon area retained; fragments below 1 square metre reported only.",
        "interpretation": "Park property area, including nonvegetated surfaces, allocated within community boundaries; ACS 2023 estimates paired with a November 2016 park inventory.",
    })
    # Avoid counting a material overlapping boundary twice. Tiny geometric
    # discrepancies are measured rather than silently erased.
    if overlap_area > city_union.area * 1e-6:
        raise ValueError("Material overlap between community boundaries")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / "communities.csv", index=False)
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": record, "geometry": mapping(transform(geom, unproject, interleaved=False))}
        for record, geom in zip(records, area_geometries)
    ]}
    _write_json(processed / "communities.geojson", geojson)
    _write_json(processed / "quality.json", quality)
    return table, area_geometries, quality
