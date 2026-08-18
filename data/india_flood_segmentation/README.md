# GeoResQ India Flood-Segmentation Dataset

This package is the **India-only satellite-image dataset** for the GeoResQ SIH prototype. It is designed for the first machine-learning capability that the prototype needs: detecting and segmenting flood or surface-water pixels from satellite imagery so that the resulting hazard mask can feed exposure analysis, safe routing, shelter scoring, and the disaster simulator.

The package uses the India event from **Sen1Floods11 v1.1**, a public georeferenced Sentinel-1/Sentinel-2 flood dataset. Only objects whose source name begins with `India_` are selected. No non-Indian image or label is included in the raw package.

## What is included

| Split | Chips | Image | Label | Intended use |
|---|---:|---|---|---|
| `train` | 467 | Sentinel-1 VV/VH, 2-channel float32 GeoTIFF | Otsu-derived weak water label | Model fitting and augmentation |
| `val` | 68 | Sentinel-1 VV/VH, 2-channel float32 GeoTIFF | Hand-labeled water/flood mask | Validation and reporting |

Each chip is **512 × 512 pixels at 10 m ground resolution**. The verified acquisition date is **2016-08-12**. The event footprint is in northeastern India, approximately `92.1506517–94.1633518°E, 24.8471365–28.2847595°N`. This is an India-specific event subset, not a claim of pan-India geographic coverage.

The image tensor has shape `(2, 512, 512)`: channel 0 is VV backscatter and channel 1 is VH backscatter, both in dB. Label values are `-1` for no-data, `0` for not-water, and `1` for water. The label schema is documented in `metadata/label_schema.csv`.

## Why this ML task matches GeoResQ

The prototype’s core flow is **Hazard → Exposure → Impact → Decision**. Satellite imagery is most directly useful for the Hazard stage. A supervised **U-Net-style semantic-segmentation model** is the recommended SIH MVP because its output is a pixel-level flood-water mask rather than a single image class. The mask can then be intersected with roads, buildings, hospitals, schools, population grids, elevation, rainfall, and shelter layers in PostGIS or GeoPandas.

The prototype’s population exposure, affected roads, hospital exposure, shelter requirements, route risk, and recommended actions should not be learned from these satellite chips alone. They should be calculated by GIS overlay and transparent rules after segmentation. A future forecasting model can use rainfall, drainage, elevation, river levels, and historical masks, but that is a second-stage model.

## Directory layout

```text
india_flood_segmentation/
├── README.md
├── raw/
│   ├── train/images/*.tif
│   ├── train/labels/*.tif
│   ├── val/images/*.tif
│   └── val/labels/*.tif
├── metadata/
│   ├── dataset_summary.csv
│   ├── label_schema.csv
│   ├── india_sen1floods11_manifest.csv
│   ├── Sen1Floods11_Metadata.geojson
│   └── india_sample_chip.png
└── scripts/
    ├── download_india_flood_dataset.py
    ├── validate_india_flood_dataset.py
    ├── inspect_geotiff_metadata.py
    ├── render_india_sample.py
    └── train_unet.py
```

## Download or reproduce the raw package

The raw GeoTIFFs are large and should be obtained through the [GitHub release asset](https://github.com/jaidhevprasad-tech/SIH/releases/tag/v1.0-india-flood-dataset) or the downloader rather than committed into ordinary Git history. From the repository root:

```bash
python3 -m pip install -r data/india_flood_segmentation/requirements.txt
python3 data/india_flood_segmentation/scripts/download_india_flood_dataset.py --workers 8
python3 data/india_flood_segmentation/scripts/validate_india_flood_dataset.py
```

The downloader queries the public Sen1Floods11 v1.1 bucket and selects only the India event. It writes `metadata/india_sen1floods11_manifest.csv`, including the source object, public source URL, local path, byte size, and SHA-256 checksum for every image and label.

## Train a baseline model

Install the requirements and run the example U-Net training script:

```bash
python3 -m pip install -r data/india_flood_segmentation/requirements.txt
python3 data/india_flood_segmentation/scripts/train_unet.py \
  --data-root data/india_flood_segmentation \
  --epochs 20 \
  --batch-size 4 \
  --output-dir runs/india_unet
```

The baseline uses masked binary cross-entropy plus Dice loss, ignores `-1` pixels, normalizes VV/VH per chip using fixed dB ranges, and reports validation IoU and Dice. Because the training masks are weakly labeled while validation masks are hand labeled, the validation score should be interpreted as an initial baseline, not as an operational accuracy guarantee.

## How to connect the model to the prototype

1. Run the U-Net model on a georeferenced India Sentinel-1 scene or chip.
2. Convert the predicted flood mask into a polygon or raster layer while retaining the CRS and acquisition timestamp.
3. Intersect the hazard layer with population, hospitals, schools, buildings, and road layers.
4. Compute exposure counts and risk zones in PostGIS.
5. Remove or penalize flooded road segments in the routing graph and rank shelters by flood risk, capacity, access, and hospital distance.
6. Pass the resulting structured fields to the Command Center, Impact, Facilities, Routes, and Simulator screens.

## India-focused expansion plan

This package provides a real India event for immediate prototyping. To move from one northeastern India event to a stronger India-wide model, add more Indian flood events from NRSC/ISRO Bhuvan or Bhoonidhi and repeat the same schema: Sentinel-1 VV/VH image, hand or authoritative flood mask, event date, WGS84 footprint, and source license. Split by **event**, not by random chip, to prevent spatial leakage. Candidate Indian regions should include Assam/Brahmaputra, Bihar, West Bengal, Odisha, Maharashtra, Kerala, Karnataka, Andhra Pradesh, and Tamil Nadu when verified event imagery and labels are available.

## Data governance and limitations

This repository does not fabricate river gauges, shelter capacities, casualties, or infrastructure damage. Such fields must be supplied by authoritative Indian sources or verified authority data. The Sen1Floods11 India event is one northeastern India event from 2016; it does not represent all Indian climates, urban forms, soil types, or disaster types. Sentinel-1 flood segmentation is therefore a hazard-mapping baseline, not a complete disaster-decision model.

## Sources and attribution

The source dataset is documented by Cloud to Street as **Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1**. Cite the original paper and retain the source attribution in derivative work. The official repository documents the v1.1 bucket, GeoTIFF structure, 10 m resolution, labels, and data access.

ISRO states that its Bhuvan and Bhoonidhi portals provide Indian Earth-observation products and disaster-management support, including flood inundation layers and a regional Sentinel-1/Sentinel-2 data hub. These portals are the preferred next sources for expanding the dataset with additional Indian events.

## References

[1]: https://github.com/cloudtostreet/Sen1Floods11 "Cloud to Street — Sen1Floods11 official repository"
[2]: https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md "Sen1Floods11 bucket documentation"
[3]: https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html "Sen1Floods11 CVPR Workshops paper"
[4]: https://www.isro.gov.in/SpaceBasedEarthObservationServices.html "ISRO Space Based Earth Observation Applications"
[5]: https://bhuvan-app1.nrsc.gov.in/disaster/disaster.php "Bhuvan NRSC Disaster Services"
[6]: https://bhoonidhi.nrsc.gov.in/ "ISRO/NRSC Bhoonidhi portal"
