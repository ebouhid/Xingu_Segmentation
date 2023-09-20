import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset.dataset import XinguDataset

# Set your constants here
BATCH_SIZE = 32
NUM_WORKERS = 20
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
INFO = 'Allbands_and_NDVI'
NUM_CLASSES = 1

compositions = {"Allbands_and_NDVI": range(1, 9)}


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, compositions, train_regions, test_regions, batch_size):
        super().__init__()
        self.compositions = compositions
        self.train_regions = train_regions
        self.test_regions = test_regions
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_ds = XinguDataset('./dataset/scenes_allbands_ndvi',
                                './dataset/truth_masks',
                                self.compositions,
                                self.train_regions,
                                PATCH_SIZE,
                                STRIDE_SIZE,
                                transforms=True)

        test_ds = XinguDataset('./dataset/scenes_allbands_ndvi',
                               './dataset/truth_masks', self.compositions,
                               self.test_regions, PATCH_SIZE, PATCH_SIZE)

        self.train_dataset = train_ds
        self.val_dataset = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=NUM_WORKERS,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=NUM_WORKERS,
                          shuffle=False)


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

    def calculate_iou(self, preds, targets, eps=1e-6):
        # Calculate intersection and union
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds) + torch.sum(targets) - intersection

        iou = intersection / (union + eps)
        return iou

    def process_outputs(self, outputs, targets):
        outputs = torch.nn.functional.interpolate(outputs,
                                                  size=targets.shape[-2:],
                                                  mode='bilinear',
                                                  align_corners=False)
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(outputs)
        return outputs

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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        out_model = self.model(inputs)
        outputs = out_model.logits
        preds = self.process_outputs(outputs, targets)
        loss = self.loss(preds, targets)
        iou = self.calculate_iou(preds, targets)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        out_model = self.model(inputs)
        outputs = out_model.logits
        preds = self.process_outputs(outputs, targets)
        preds = (preds > 0.5).float()
        loss = self.loss(preds, targets)
        iou = self.calculate_iou(preds, targets)
        accuracy = (preds == targets).float().mean()
        precision = (preds * targets).sum() / (preds.sum() + 1e-6)
        recall = (preds * targets).sum() / (targets.sum() + 1e-6)
        fscore = 2 * precision * recall / (precision + recall + 1e-6)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        self.log('val_fscore', fscore, on_epoch=True)


train_regions = [1, 2, 5, 6, 7, 8, 9, 10]
test_regions = [3, 4]

for COMPOSITION in compositions:
    CHANNELS = len(compositions[COMPOSITION])

    configs = [(SegformerForSemanticSegmentation(
        SegformerConfig(num_channels=CHANNELS, num_labels=NUM_CLASSES)),
                torch.nn.BCEWithLogitsLoss(), 1e-3)]

    for (model, loss, lr) in configs:
        model = SegmentationModel(model, loss, lr, COMPOSITION, train_regions,
                                  test_regions, BATCH_SIZE)
        # run_name = f'{COMPOSITION}_{model.__class__.__name__}'
        logger = pl.loggers.MLFlowLogger(experiment_name=INFO)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=NUM_EPOCHS,
        )
        trainer.fit(model,
                    datamodule=SegmentationDataModule(
                        compositions[COMPOSITION], train_regions, test_regions,
                        BATCH_SIZE))
