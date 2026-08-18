from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'data' / 'india_flood_segmentation'
chip = '1029501'
image = tifffile.imread(root / 'raw' / 'train' / 'images' / f'{chip}.tif').astype('float32')
label = tifffile.imread(root / 'raw' / 'train' / 'labels' / f'{chip}.tif')
mask = np.where(label == 1, 1.0, np.nan)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, data, title in [
    (axes[0], image[0], 'VV backscatter (dB)'),
    (axes[1], image[1], 'VH backscatter (dB)'),
    (axes[2], mask, 'Weak flood label'),
]:
    if title == 'Weak flood label':
        ax.imshow(data, cmap='Blues', vmin=0, vmax=1)
    else:
        ax.imshow(data, cmap='gray', vmin=-35, vmax=5)
    ax.set_title(title)
    ax.axis('off')
fig.suptitle(f'GeoResQ India flood chip {chip} — Sen1Floods11 India event')
out = root / 'metadata' / 'india_sample_chip.png'
fig.savefig(out, dpi=180)
print(out)
