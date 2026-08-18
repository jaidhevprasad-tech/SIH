# SIH — Geospatial Platform for Disaster Information and Decision Support

This repository contains the foundation for a disaster-information and decision-support platform focused on **India**, with an Assam deployment package and an India-only satellite flood-segmentation package for the GeoResQ prototype.

## India-only satellite ML package

The primary satellite training package is [`data/india_flood_segmentation/`](data/india_flood_segmentation/). It contains 535 real Indian Sentinel-1 SAR chips from the Sen1Floods11 India event: 467 weakly labeled training chips and 68 hand-labeled validation chips. The package includes the downloader, SHA-256 manifest, validator, sample visualization, dataset documentation, and a runnable PyTorch U-Net baseline. Read [`data/india_flood_segmentation/README.md`](data/india_flood_segmentation/README.md) before training.

The raw GeoTIFFs are intentionally distributed as a GitHub release asset and can also be reproduced from the public source bucket with the included downloader. The model output is a flood-water hazard mask; population, hospitals, schools, buildings, roads, routes, shelters, and recommended actions remain GIS overlay and decision-support stages rather than labels learned from satellite chips alone.

The Assam package remains available for the Assam-specific data catalog and weather feature workflow. It should be treated as an operational data-integration package, while the India satellite package is the supervised computer-vision training core.

## What the Assam dataset package contains

| Archive path | Contents | Role in the platform |
| :--- | :--- | :--- |
| `data/assam/boundaries/assam_boundary.geojson` | Assam area-of-interest boundary in WGS 84 (EPSG:4326), sourced from OpenStreetMap contributors | Clip weather, satellite, and exposure layers to Assam. |
| `data/assam/metadata/dataset_catalog.csv` | Source catalogue for CWC, ASDMA, NASA GPM, Copernicus Sentinel, WorldPop, OpenStreetMap, and Assam GIS GeoHUB data | Data governance, refresh schedule, licensing, and model mapping. |
| `data/assam/metadata/emergency_facilities_template.csv` | Validated schema for authorized shelter, relief camp, medical facility, and resource data | Safe local-authority data intake. |
| `data/assam/scripts/export_assam_earth_engine.js` | Google Earth Engine workflow for GPM rainfall, Sentinel-1 SAR, Sentinel-2 optical, and WorldPop exports | XGBoost rainfall inputs, U-Net flood-mapping inputs, and exposure layers. |
| `data/assam/README.md` | Full dataset guide with source links, licences, operational safeguards, and setup instructions | Project documentation. |
| `database/assam_postgis.sql` | Assam-specific PostGIS schema and transparent risk/exposure view | Flood extent, exposure, shelter, resource, and Risk Engine storage. |

## Architecture

```text
Data (Weather + Satellite + GIS)
        ↓
XGBoost (forecasting) + U-Net (flood extent) + PostGIS (exposure)
        ↓
Risk Engine (hazard score + exposure score)
        ↓
Risk Score
        ↓
Evacuation + Shelters + Resources
        ↓
GIS Dashboard
```

## Important operational safeguards

The package contains **no fabricated river-gauge, relief-camp, shelter-capacity, or flood-event records**. Dynamic satellite and weather data must be obtained at run time from their authoritative sources, with source time and provenance retained. Any emergency-facility record must be supplied by an authorized authority and currently verified before operational use.

## Start Here

1. Download and extract [`assam_dataset_package.zip`](assam_dataset_package.zip).
2. Read `data/assam/README.md` for exact data sources, terms, refresh schedules, and setup steps.
3. Run `database/assam_postgis.sql` on a PostGIS-enabled database.
4. Open `data/assam/scripts/export_assam_earth_engine.js` in Google Earth Engine and set the event dates before exporting data.
