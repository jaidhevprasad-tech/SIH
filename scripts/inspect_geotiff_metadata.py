from pathlib import Path
import numpy as np

sample = Path(__file__).resolve().parents[1] / 'data' / 'india_flood_segmentation' / 'raw' / 'train' / 'images'
paths = sorted(sample.glob('*.tif'))[:1]
if not paths:
    raise SystemExit('No sample files found')
path = paths[0]
print('sample=', path)
for module_name in ['tifffile', 'osgeo.gdal']:
    try:
        module = __import__(module_name.split('.')[0])
        print(module_name, 'available')
    except Exception as exc:
        print(module_name, 'missing', type(exc).__name__)
try:
    import tifffile
    arr = tifffile.imread(path)
    print('tifffile_shape=', arr.shape)
    print('tifffile_dtype=', arr.dtype)
    print('tifffile_minmax=', float(np.nanmin(arr)), float(np.nanmax(arr)))
except Exception as exc:
    print('tifffile_read_error=', type(exc).__name__, str(exc))
