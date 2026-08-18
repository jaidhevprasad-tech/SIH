#!/usr/bin/env python3
"""Download the India-only GeoResQ flood-segmentation subset.

The script queries the public Sen1Floods11 v1.1 Google Cloud Storage API and
intentionally selects only objects whose names contain the India event prefix.
It downloads Sentinel-1 VV/VH images and paired flood masks:

* train: weakly labeled S1Weak + S1OtsuLabelWeak (467 chips)
* val: hand-labeled S1Hand + LabelHand (68 chips)

No non-Indian object is selected by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
BUCKET = "https://storage.googleapis.com/sen1floods11/"
ROOT = Path(__file__).resolve().parents[1] / "data" / "india_flood_segmentation"

LAYERS = {
    "train": {
        "image_prefix": "v1.1/data/flood_events/WeaklyLabeled/S1Weak/India_",
        "label_prefix": "v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/India_",
        "image_layer": "S1Weak",
        "label_layer": "S1OtsuLabelWeak",
    },
    "val": {
        "image_prefix": "v1.1/data/flood_events/HandLabeled/S1Hand/India_",
        "label_prefix": "v1.1/data/flood_events/HandLabeled/LabelHand/India_",
        "image_layer": "S1Hand",
        "label_layer": "LabelHand",
    },
}


def list_objects(prefix: str) -> list[dict]:
    objects: list[dict] = []
    token = None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        response = requests.get(API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        objects.extend(payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            return sorted(objects, key=lambda item: item["name"])


def chip_id(name: str) -> str:
    return Path(name).name.split("_")[1]


def download_one(item: dict, out_path: Path, force: bool = False) -> tuple[int, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item.get("size", 0))
    if out_path.exists() and not force and out_path.stat().st_size == expected:
        return expected, hashlib.sha256(out_path.read_bytes()).hexdigest()
    url = BUCKET + quote(item["name"], safe="/")
    temp = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with temp.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
    actual = temp.stat().st_size
    if expected and actual != expected:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {item['name']}: expected {expected}, got {actual}")
    temp.replace(out_path)
    return actual, hashlib.sha256(out_path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, str, str, dict, Path]] = []
    for split, config in LAYERS.items():
        image_items = {chip_id(x["name"]): x for x in list_objects(config["image_prefix"])}
        label_items = {chip_id(x["name"]): x for x in list_objects(config["label_prefix"])}
        if set(image_items) != set(label_items):
            raise RuntimeError(f"Image/label IDs do not match for {split}")
        for cid in sorted(image_items):
            jobs.append((split, cid, "image", config["image_layer"], image_items[cid], ROOT / "raw" / split / "images" / f"{cid}.tif"))
            jobs.append((split, cid, "label", config["label_layer"], label_items[cid], ROOT / "raw" / split / "labels" / f"{cid}.tif"))

    manifest_path = ROOT / "metadata" / "india_sen1floods11_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str], tuple[int, str]] = {}
    if not args.manifest_only:
        print(f"Downloading/validating {len(jobs)} India-only objects")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {
                pool.submit(download_one, item, path, args.force): (split, cid, role, layer, item, path)
                for split, cid, role, layer, item, path in jobs
            }
            for index, future in enumerate(as_completed(pending), start=1):
                split, cid, role, layer, item, path = pending[future]
                size, digest = future.result()
                results[(split, str(path.relative_to(ROOT)))] = (size, digest)
                print(f"[{index}/{len(jobs)}] {split}/{role}/{cid}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "chip_id", "role", "source_layer", "source_object", "source_url", "local_path", "source_size_bytes", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, cid, role, layer, item, path in jobs:
            relative = str(path.relative_to(ROOT))
            if path.exists():
                size = path.stat().st_size
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                size = int(item.get("size", 0))
                digest = ""
            writer.writerow({
                "split": split,
                "chip_id": cid,
                "role": role,
                "source_layer": layer,
                "source_object": item["name"],
                "source_url": BUCKET + quote(item["name"], safe="/"),
                "local_path": relative,
                "source_size_bytes": size,
                "sha256": digest,
            })
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
