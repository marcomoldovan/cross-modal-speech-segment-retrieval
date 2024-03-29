import os
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, concatenate_datasets
from pytorch_lightning import LightningDataModule

from src.datamodules.components.librispeech_dataset import LibriSpeechDataset
from src.utils import get_logger

log = get_logger(__name__)


class LibriSpeechDataModule(LightningDataModule):
    """
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
        cache_dir,
        train_batch_size,
        val_batch_size,
        test_batch_size,
        load_preprocessed_data=False,
        train_split='train.360',
        pin_memory=True,
        debug=False
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()
        
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
            
        self.libri_train: Optional[Dataset] = None
        self.libri_val: Optional[Dataset] = None
        self.libri_test: Optional[Dataset] = None
        
        
    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        
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
            if not os.path.isdir(self.hparams.cache_dir):
                log.info("Downloading LibriSpeech dataset...")
                load_dataset('librispeech_asr', split='train.100', cache_dir=self.hparams.cache_dir)
            load_dataset('librispeech_asr', 'clean', split=self.hparams.train_split, cache_dir=self.hparams.cache_dir)
            
        
    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""
        
        # Assign train/val datasets for use in dataloaders
        
        if stage == "fit" or stage is None:
            if self.hparams.load_preprocessed_data:
                if not self.hparams.debug:
                    libri_shards_list = []
                    for i in range(len(next(os.walk(self.hparams.data_dir))[1])-1):
                        loaded_libri_shard = load_from_disk(f"{self.hparams.data_dir}/{i}/")
                        libri_shards_list.append(loaded_libri_shard)
                    self.libri_train = concatenate_datasets(libri_shards_list)
                    self.libri_val = LibriSpeechDataset(load_from_disk(f'{self.hparams.data_dir}/{len(next(os.walk(self.hparams.data_dir))[1])-1}'))
                else:
                    self.libri_train = LibriSpeechDataset(load_from_disk(f'{self.hparams.data_dir}/0'))
                    self.libri_val = LibriSpeechDataset(load_from_disk(f'{self.hparams.data_dir}/1'))

                # self.libri_train = LibriSpeechDataset(load_from_disk(f'{self.hparams.data_dir}/{self.hparams.split}/'))
                # self.libri_val = LibriSpeechDataset(load_from_disk(f'{self.hparams.data_dir}/validation/'))
            else:
                self.libri_train = LibriSpeechDataset(load_dataset('librispeech_asr', 'clean', split=self.hparams.train_split, cache_dir=self.hparams.cache_dir))
                self.libri_val = LibriSpeechDataset(load_dataset('librispeech_asr', 'clean', split='validation', cache_dir=self.hparams.cache_dir))
                
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            if self.hparams.load_preprocessed_data:
                self.libri_test = LibriSpeechDataset(load_from_disk(f'{self.hparams.data_dir}/test/'))
            else:
                self.libri_test = LibriSpeechDataset(load_dataset('librispeech_asr', 'clean', split='test', cache_dir=self.hparams.cache_dir))
        
        if stage == "predict" or stage is None:
            raise Exception("""This DataModule is not designed to be used for prediction.
                            Please use the Spotify DataModule for prediction.""")
    
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.libri_train, 
            batch_size=self.hparams.train_batch_size, 
            shuffle=True, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.libri_val, 
            batch_size=self.hparams.val_batch_size, 
            shuffle=False, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.libri_test, 
            batch_size=self.hparams.test_batch_size, 
            shuffle=False, 
            collate_fn=self.collator, 
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory
            )
        
