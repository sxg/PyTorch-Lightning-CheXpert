from CheXpertModule import CheXpertModule
from CheXpertDataModule import CheXpertDataModule
from lightning.pytorch.cli import LightningCLI


def main():
    LightningCLI(CheXpertModule, CheXpertDataModule)


if __name__ == "__main__":
    main()
