from skimage import io
from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def band_norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min) / (band_max - band_min))


loop = tqdm(range(1, 11))

for i in loop:
    band_paths = sorted(
        os.listdir(f'xingu/LS/AE-X{i :02d}/LS-AE-X{i :02d}_8b'))
    bands = []
    for fname in band_paths:
        with rasterio.open(
                os.path.join(f'xingu/LS/AE-X{i :02d}/LS-AE-X{i :02d}_8b',
                             fname)) as src:
            band = src.read(1)
            bands.append(band)

    landsat_data = np.dstack(bands)

    np.save(f'allbands_scenes/allbands_x{i :02d}.npy', landsat_data)
