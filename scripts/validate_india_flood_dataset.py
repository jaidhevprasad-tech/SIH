#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "india_flood_segmentation"
MANIFEST = ROOT / "metadata" / "india_sen1floods11_manifest.csv"


def read_tif(path: Path):
    with Image.open(path) as img:
        frames = []
        try:
            n_frames = getattr(img, "n_frames", 1)
        except Exception:
            n_frames = 1
        for frame in range(n_frames):
            img.seek(frame)
            frames.append(np.array(img))
        arr = np.stack(frames, axis=0) if len(frames) > 1 else frames[0]
        return arr


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError("Manifest is empty")
    by_key = {(r["split"], r["chip_id"], r["role"]): r for r in rows}
    errors = []
    stats = Counter()
    label_values = Counter()

    for row in rows:
        path = ROOT / row["local_path"]
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        actual = path.stat().st_size
        expected = int(row["source_size_bytes"])
        if actual != expected:
            errors.append(f"size mismatch: {path} expected={expected} actual={actual}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            errors.append(f"hash mismatch: {path}")
        try:
            arr = read_tif(path)
        except Exception as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        stats[(row["split"], row["role"], str(arr.shape))] += 1
        if row["role"] == "label":
            vals, counts = np.unique(arr, return_counts=True)
            for value, count in zip(vals.tolist(), counts.tolist()):
                label_values[int(value)] += int(count)
            allowed = {-1, 0, 1, 255}
            bad = set(int(x) for x in vals.tolist()) - allowed
            if bad:
                errors.append(f"unexpected label values in {path}: {sorted(bad)}")

    image_ids = {(r["split"], r["chip_id"]) for r in rows if r["role"] == "image"}
    label_ids = {(r["split"], r["chip_id"]) for r in rows if r["role"] == "label"}
    if image_ids != label_ids:
        errors.append(f"image/label mismatch: images={len(image_ids)} labels={len(label_ids)}")

    print("manifest_rows=", len(rows))
    print("image_chips=", len(image_ids))
    print("label_chips=", len(label_ids))
    print("shape_stats=")
    for key, value in sorted(stats.items()):
        print(" ", key, value)
    print("label_values=", dict(sorted(label_values.items())))
    print("errors=", len(errors))
    for error in errors[:25]:
        print("ERROR", error)
    if errors:
        return 1
    print("VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
