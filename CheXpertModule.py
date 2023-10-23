from torch import nn, optim
import lightning.pytorch as pl
from torchvision import models
from torchvision.utils import make_grid
from torchmetrics import Accuracy, AUROC


class CheXpertModule(pl.LightningModule):
    def __init__(self, tasks, lr):
        super().__init__()

        self.tasks = tasks
        self.lr = lr
        self.save_hyperparameters()

        # Setup the model, loss function, and other metrics
        self.model = models.densenet121(weights="DEFAULT")
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, len(self.tasks)),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.BCELoss()
        self.accuracy_fn = Accuracy(
            task="multilabel", num_labels=len(self.tasks)
        )
        self.auroc_fn = AUROC(task="multilabel", num_labels=len(self.tasks))

        # Flags for logging
        self.train_logged_images = False
        self.val_logged_images = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=self.lr)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self(imgs)
        loss = self.loss_fn(output, labels)
        self.log("train/loss", loss)

        # Log sample data to TensorBoard
        if not self.train_logged_images:
            self.train_logged_images = True
            preds = (output > 0.5).int()
            img_grid = make_grid(imgs, nrow=4)
            self.logger.experiment.add_image(
                "train/imgs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "train/labels",
                CheXpertModule._format_labels(labels, self.tasks),
                self.current_epoch,
            )
            self.logger.experiment.add_text(
                "train/preds",
                CheXpertModule._format_labels(preds, self.tasks),
                self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        output = self(imgs)
        loss = self.loss_fn(output, labels)
        self.log("val/loss", loss)
        acc = self.accuracy_fn(output, labels)
        self.log("val/acc", acc)
        auroc = self.auroc_fn(output, labels.int())
        self.log("val/auroc", auroc)

        # Log sample data to TensorBoard
        if not self.val_logged_images:
            self.val_logged_images = True
            preds = (output > 0.5).int()
            img_grid = make_grid(imgs, nrow=4)
            self.logger.experiment.add_image(
                "val/imgs", img_grid, self.current_epoch
            )
            self.logger.experiment.add_text(
                "val/labels",
                CheXpertModule._format_labels(labels, self.tasks),
                self.current_epoch,
            )
            self.logger.experiment.add_text(
                "val/preds",
                CheXpertModule._format_labels(preds, self.tasks),
                self.current_epoch,
            )

        return {"val/loss": loss, "val/acc": acc, "val/auroc": auroc}

    def on_training_epoch_end(self):
        self.train_logged_images = False

    def on_validation_epoch_end(self):
        self.val_logged_images = False

    def _format_labels(labels, tasks):
        string = " ".join(tasks) + "\n"
        for idx in range(labels.shape[0]):
            string += labels[idx, :] + "\n"
        return string
