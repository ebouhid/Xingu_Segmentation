from torch.utils.data import Dataset
import os
import numpy as np
import warnings


class XinguDataset(Dataset):
    def __init__(self, scenes_dir, masks_dir, composition, transforms=None):
        self.composition = composition

        self.img_path = scenes_dir
        self.msk_path = masks_dir

        self.image_paths = sorted(os.listdir(self.img_path))
        # print(f'total paths: {len(self.image_paths)}')
        # print(f'paths: {self.image_paths}')
        self.mask_paths = sorted(os.listdir(self.msk_path))

        self.images = []
        self.masks = []

        self.transforms = transforms
        self.angle_inc = 45

        # load scenes
        for img_scene in self.image_paths:
            self.images.append(np.load(os.path.join(self.img_path, img_scene)))

        for msk_scene in self.mask_paths:
            self.masks.append(
                np.load(os.path.join(self.msk_path, msk_scene)).squeeze())

        if self.transforms:
            warnings.warn('Transforms are not implemented yet.', UserWarning)

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = image.transpose(2, 0, 1)

        # create composition
        nbands = len(self.composition)
        combination = np.zeros((nbands, ) + image.shape[1:])
        # print(f'combination shape: {combination.shape}')

        for i in range(nbands):
            combination[i, :, :] = image[(self.composition[i] - 1), :, :]

        combination = np.float32(combination) / 255

        mask = np.expand_dims(mask, axis=0)
        mask = np.float32(mask)

        combination = combination.astype(np.float32)

        return combination, mask
