import os
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.datamodules.components.spokensquad_dataset import SpokenSQuADDataset, SpokenSQuADEmbeddedDataset
from src.utils import get_logger

log = get_logger(__name__)


class SpokenSQuADDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        collator,
        data_dir,
        train_batch_size,
        val_batch_size,
        test_batch_size,
        load_preprocessed_data=False,
        pin_memory=True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        
        self.collator = collator
        
        # on windows we need to set num_workers to 0 due to a bug
        if os.name == 'nt':
            self.num_workers = 0
        else:
            self.num_workers = os.cpu_count()
            
        if self.hparams.load_preprocessed_data:
            self.num_proc = 1
        else:
            self.num_proc = os.cpu_count()

        self.spokensquad_train: Optional[Dataset] = None
        self.spokensquad_val: Optional[Dataset] = None
        self.spokensquad_test: Optional[Dataset] = None


    def prepare_data(self):
        """Download data if needed.
        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # preprocessed meaning that the input values were already fed through the convolutional feature extractor
        if self.hparams.load_preprocessed_data:
            log.info("Loading preembedded text and speech feature from conv module.")
            log.warning(
                """Will NOT run any preprocessing from datamodule as \
                of right now. Make sure to run preprocess.py from \
                the root directory first. Set corresponding directories\
                data_dir for saving and loading data  in the config correctly.
                """
                )
            assert os.path.exists(self.hparams.data_dir), "Preembedded dataset does not exist, run preprocessing first."
        else:
            if os.path.isdir(self.hparams.data_dir):
                log.info("Preembedded dataset exists, will not download again.")
            else:
                log.info("Preembedded dataset does not exist, will download.")
                self.download_data()


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        
        # Assign train/val datasets for use in dataloaders
        
        if stage == "fit" or stage is None:
            if self.hparams.load_preprocessed_data:
                self.spokensquad_train = SpokenSQuADEmbeddedDataset()
                self.spokensquad_val = SpokenSQuADEmbeddedDataset()
            else:
                self.spokensquad = SpokenSQuADDataset(f'{self.hparams.data_dir}/spoken_train-v1.1.json')
                self.spokensquad_train, self.spokensquad_val = random_split(
                    self.spokensquad, 
                    [int(0.9 * len(self.spokensquad)), len(self.spokensquad) - int(0.9 * len(self.spokensquad))],
                    generator=torch.Generator().manual_seed(42)
                    )
                
        if stage == "test" or stage is None:
            if self.hparams.load_preprocessed_data:
                self.spokensquad_test = SpokenSQuADEmbeddedDataset()
            else:
                self.spokensquad_test = SpokenSQuADDataset(f'{self.hparams.data_dir}/spoken_test-v1.1_WER54.json')
        
        if stage == "predict" or stage is None:
            raise Exception("""This DataModule is not designed to be used for prediction.
                            Please use the Spotify DataModule for prediction.""")


    def train_dataloader(self):
        return DataLoader(
            dataset=self.spokensquad_train, 
            batch_size=self.hparams.train_batch_size, 
            shuffle=True, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.spokensquad_val, 
            batch_size=self.hparams.val_batch_size, 
            shuffle=False, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.spokensquad_test, 
            batch_size=self.hparams.test_batch_size, 
            shuffle=False, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )