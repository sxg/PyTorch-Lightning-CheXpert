from model import CheXpertModule
from data import CheXpertDataModule
from lightning.pytorch.cli import LightningCLI


def main():
    LightningCLI(CheXpertModule, CheXpertDataModule)


if __name__ == "__main__":
    main()
