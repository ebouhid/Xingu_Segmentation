import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset.dataset import XinguDataset

# Set your constants here
BATCH_SIZE = 32
NUM_EPOCHS = 10
PATCH_SIZE = 256
STRIDE_SIZE = 64
INFO = 'ShuffleTest_PytorchLightning'
NUM_CLASSES = 1


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, loss, lr, compositions, train_regions,
                 test_regions, batch_size):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.compositions = compositions
        self.train_regions = train_regions
        self.test_regions = test_regions
        self.batch_size = batch_size

    def on_fit_start(self):
        self.logger.log_hyperparams({
            "model": self.model.__class__.__name__,
            "loss": self.loss.__class__.__name__,
            "lr": self.lr,
            "composition": self.compositions,
            "batch_size": self.batch_size,
            "num_epochs": NUM_EPOCHS,
            "patch_size": PATCH_SIZE,
            "stride_size": STRIDE_SIZE,
            "Description": INFO,
            "train_regions": self.train_regions,
            "test_regions": self.test_regions
        })

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        train_ds = XinguDataset('./dataset/scenes_allbands',
                                './dataset/truth_masks',
                                self.compositions,
                                self.train_regions,
                                PATCH_SIZE,
                                STRIDE_SIZE,
                                transforms=True)
        return DataLoader(train_ds,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=True)

    def val_dataloader(self):
        test_ds = XinguDataset('./dataset/scenes_allbands',
                               './dataset/truth_masks', self.compositions,
                               self.test_regions, PATCH_SIZE, STRIDE_SIZE)
        return DataLoader(test_ds,
                          batch_size=self.batch_size,
                          drop_last=True,
                          shuffle=False)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        metrics = smp.utils.metrics.IoU()(outputs, targets)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_iou', metrics, on_epoch=True)


compositions = {
    "6": [6],
    "65": [6, 5],
    "651": [6, 5, 1],
}

train_regions = [1, 2, 5, 6, 7, 8, 9, 10]
test_regions = [3, 4]

for COMPOSITION in compositions:
    CHANNELS = len(compositions[COMPOSITION])

    configs = [(smp.DeepLabV3Plus(in_channels=CHANNELS,
                                  classes=NUM_CLASSES,
                                  activation='sigmoid'),
                smp.utils.losses.JaccardLoss(), 1e-3)]

    for (model, loss, lr) in configs:
        model = SegmentationModel(model, loss, lr, compositions[COMPOSITION],
                                  train_regions, test_regions, BATCH_SIZE)
        run_name = f'{COMPOSITION}_NoShuffle'
        logger = pl.loggers.MLFlowLogger(experiment_name=INFO,
                                         run_name=run_name)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=NUM_EPOCHS,
        )
        trainer.fit(model)
