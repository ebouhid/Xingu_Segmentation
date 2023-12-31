import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
import mlflow
from dataset.dataset import XinguDataset
import time
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

BATCH_SIZE = 64
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
INFO = 'WithTransforms'
NUM_CLASSES = 1

os.environ['MLFLOW_EXPERIMENT_NAME'] = 'RecentVsAll'

compositions = {
    "6": [6],
    "65": [6, 5],
    "651": [6, 5, 1],
    "6513": [6, 5, 1, 3],
    "6514": [6, 5, 1, 4],
    "6517": [6, 5, 1, 7],
    "All+NDVI": range(1, 9),
    "RGB": [4, 3, 2]
}

train_regions = [1, 2, 6, 7, 8, 9, 10]
test_regions = [3, 4]
for COMPOSITION in compositions:
    CHANNELS = len(compositions[COMPOSITION])
    # (model, loss, lr)
    configs = [(smp.DeepLabV3Plus(in_channels=CHANNELS,
                                  classes=NUM_CLASSES,
                                  activation='sigmoid'),
                smp.utils.losses.BCELoss(), 5e-4)]
    for (model, loss, lr) in configs:
        best_epoch = 0
        max_f1 = 0
        max_precision = 0
        max_iou = 0
        max_accuracy = 0
        max_recall = 0

        print(f"{10 * '#'} {model.__class__.__name__} {10*'#'}")

        # instantiating datasets
        # Define transform pipeline
        transforms = A.Compose([
            A.Resize(256, 256),
            A.Rotate(limit=90, p=0.8),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomBrightnessContrast(),
            ToTensorV2(),
        ])
        train_ds = XinguDataset('./dataset/selected_patches/train/images',
                                './dataset/selected_patches/train/masks',
                                compositions[COMPOSITION],
                                transforms=transforms)
        test_ds = XinguDataset('./dataset/selected_patches/test/images',
                               './dataset/selected_patches/test/masks',
                               compositions[COMPOSITION],
                               transforms=None)

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=lr),
        ])

        metrics = [
            smp.utils.metrics.IoU(),
            smp.utils.metrics.Fscore(),
            smp.utils.metrics.Precision(),
            smp.utils.metrics.Accuracy(),
            smp.utils.metrics.Recall()
        ]

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device='cuda',
            verbose=True,
        )
        test_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device='cuda',
            verbose=True,
        )

        # dataloaders for this fold
        train_dataloader = torch.utils.data.DataLoader(dataset=train_ds,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=8)

        test_dataloader = torch.utils.data.DataLoader(dataset=test_ds,
                                                      batch_size=8,
                                                      shuffle=False,
                                                      num_workers=8)
        with mlflow.start_run():
            # logging parameters
            mlflow.log_params({
                "model": model.__class__.__name__,
                "loss": loss.__class__.__name__,
                "lr": lr,
                "composition": COMPOSITION,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "patch_size": PATCH_SIZE,
                "stride_size": STRIDE_SIZE,
                "Description": INFO,
                "train_regions": train_regions,
                "test_regions": test_regions
            })

            start = time.time()

            for epoch in range(1, NUM_EPOCHS + 1):
                print(f'\nEpoch: {epoch}')
                train_logs = train_epoch.run(train_dataloader)
                test_logs = test_epoch.run(test_dataloader)

                if max_iou < test_logs['iou_score']:
                    torch.save(
                        model,
                        f'./models/{INFO}-{model.__class__.__name__}-{loss.__class__.__name__}-{COMPOSITION}.pth'
                    )
                    torch.save(
                        model.state_dict(),
                        f'./models/{INFO}-{model.__class__.__name__}-{loss.__class__.__name__}-{COMPOSITION}-StateDict.pth'
                    )

                    max_iou = test_logs['iou_score']
                    max_precision = test_logs['precision']
                    max_f1 = test_logs['fscore']
                    max_accuracy = test_logs['accuracy']
                    max_recall = test_logs['recall']
                    best_epoch = epoch
                    print('Model saved!')

                # gathering data
                loss_train = next(iter(train_logs.values()))
                iou_score_train = train_logs['iou_score']

                precision_test = test_logs['precision']
                f1_test = test_logs['fscore']
                iou_score_test = test_logs['iou_score']
                accuracy_test = test_logs['accuracy']
                recall_test = test_logs['recall']

                # logging to mlflow
                mlflow.log_metric('train_loss', loss_train, epoch)
                mlflow.log_metric('train_iou', iou_score_train, epoch)
                mlflow.log_metric('test_precision', precision_test, epoch)
                mlflow.log_metric('test_f1', f1_test, epoch)
                mlflow.log_metric('test_iou', iou_score_test, epoch)
                mlflow.log_metric('test_accuracy', accuracy_test, epoch)
                mlflow.log_metric('test_recall', recall_test, epoch)

            end = time.time()
            execution_time = end - start

            # Convert execution time to minutes and seconds
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)

            with open(
                    f'results/results-{INFO}-{model.__class__.__name__}-{lr :.0e}-{COMPOSITION}.txt',
                    'w') as f:
                f.write("TEST RESULTS\n")
                f.write(f'{model.__class__.__name__}-{COMPOSITION}\n')
                f.write(f'Precision: {max_precision :.4f}\n')
                f.write(f'F1 Score: {max_f1 :.4f}\n')
                f.write(f'IoU: {max_iou :.4f}\n')
                f.write(f'Accuracy: {max_accuracy :.4f}\n')
                f.write(f'Recall: {max_recall :.4f}\n')
                f.write(f'On epoch: {best_epoch}\n')
                f.write(f'Time: {minutes}m {seconds}s\n')
                f.write(f'Loss: {loss.__class__.__name__}\n')
