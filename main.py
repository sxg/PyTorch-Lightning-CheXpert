from CheXpertModule import CheXpertModule
from CheXpertDataModule import CheXpertDataModule
import lightning.pytorch as pl


def main():
    tasks = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    model = CheXpertModule(tasks, 1e-4)
    data = CheXpertDataModule(tasks)
    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
