import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset.dataset import XinguDataset

# Set your constants here
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
INFO = 'GeneticCombinations'
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

    def calculate_iou(self, preds, targets, eps=1e-6):
        # Calculate intersection and union
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds) + torch.sum(targets) - intersection

        iou = intersection / (union + eps)
        return iou
    
    def process_outputs(self, outputs, targets):
        outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
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
                          num_workers=12,
                          shuffle=True)

    def val_dataloader(self):
        test_ds = XinguDataset('./dataset/scenes_allbands',
                               './dataset/truth_masks', self.compositions,
                               self.test_regions, PATCH_SIZE, STRIDE_SIZE)
        return DataLoader(test_ds,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=12,
                          shuffle=False)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        out_model = self.model(inputs)
        outputs = out_model.logits
        preds = self.process_outputs(outputs, targets)     
        loss = self.loss(preds, targets)
        iou = self.calculate_iou(preds, targets)
        self.log('train_iou', iou)
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
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_iou', iou, on_epoch=True)



compositions = {
    "6": [6],
    "65": [6, 5],
    "651": [6, 5, 1],
    "6513": [6, 5, 1, 3],
    "6514": [6, 5, 1, 4],
    "6517": [6, 5, 1, 7]
}

train_regions = [1, 2, 5, 6, 7, 8, 9, 10]
test_regions = [3, 4]

for COMPOSITION in compositions:
    CHANNELS = len(compositions[COMPOSITION])

    configs = [(SegformerForSemanticSegmentation(SegformerConfig(num_channels=CHANNELS, num_labels=NUM_CLASSES)),
                torch.nn.BCEWithLogitsLoss(), 1e-3)]

    for (model, loss, lr) in configs:
        model = SegmentationModel(model, loss, lr, compositions[COMPOSITION],
                                  train_regions, test_regions, BATCH_SIZE)
        # run_name = f'{COMPOSITION}_{model.__class__.__name__}'
        logger = pl.loggers.MLFlowLogger(experiment_name=INFO)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=NUM_EPOCHS,
        )
        trainer.fit(model)
