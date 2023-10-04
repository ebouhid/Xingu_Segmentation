import os
import glob
import numpy as np


def cut_array_into_subpatches(image, mask):
    image_patches = []
    mask_patches = []

    patch_size = 256
    stride = 64
    small_stride = 5
    i = 0
    j = 0
    print(f'{40*"#"} newcall {40*"#"}')
    while i < image.shape[0] - patch_size + 1:
        while j < image.shape[1] - patch_size + 1:
            image_patch = image[i:i + patch_size, j:j + patch_size]
            mask_patch = mask[i:i + patch_size, j:j + patch_size]
            print(
                f'Analyzing patch {i}:{i + patch_size}, {j}:{j + patch_size}')
            if np.any(mask_patch == 1):
                image_patches.append(image_patch)
                mask_patches.append(mask_patch)
                stride = small_stride
                break
            else:
                stride = 64
            j += stride
        i += stride
    print(f"Found {len(image_patches)} patches")
    return {"image_patches": image_patches, "mask_patches": mask_patches}


def get_region(path):
    return path.split('/')[-1].split('_')[-1].split('.')[0]


if __name__ == '__main__':
    train_regions = ['x01', 'x02', 'x06', 'x07', 'x08', 'x09', 'x10']
    test_regions = ['x03', 'x04']
    image_paths = sorted(glob.glob('scenes_allbands_ndvi/*'))
    mask_paths = sorted(glob.glob('truth_masks/*'))

    print(f"Found {len(image_paths)} images")
    print(f"Found {len(mask_paths)} masks")

    for image_path, mask_path in zip(image_paths, mask_paths):
        image_region = get_region(image_path)
        mask_region = get_region(mask_path)

        assert image_region == mask_region

        image = np.load(image_path)
        print(f'Image for region {image_region} loaded')
        mask = np.load(mask_path)
        print(f'Mask for region {mask_region} loaded')
        patches = cut_array_into_subpatches(image, mask)
        for i, (image_patch, mask_patch) in enumerate(
                zip(patches['image_patches'], patches['mask_patches'])):
            if image_region in train_regions:
                scope = 'train'
            else:
                scope = 'test'
            np.save(
                f'selected_patches/{scope}/images/{get_region(image_path)}_{i}.npy',
                image_patch)
            np.save(
                f'selected_patches/{scope}/masks/{get_region(mask_path)}_{i}.npy',
                mask_patch)
