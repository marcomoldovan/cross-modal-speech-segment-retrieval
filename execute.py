import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from models import ParallelSpeechAndTextModel
from datamodules import LibriSpeechDataModule
from callbacks import LibriLoggingCallback
from utils import build_run_name



def main():
  
  # ---------------------
  # non-essential
  # ---------------------
  
  #TODO log artifacts and store them on GDrive or wandb cloud
  run_name = build_run_name()
  wandb_logger = WandbLogger(name=run_name, project="cross-modal-speech-segment-retrieval", log_model='all')
  libri_logging_callback = LibriLoggingCallback()
  
  # ---------------------
  # data
  # ---------------------

  libri_data_module = LibriSpeechDataModule()
  # libri_data_module.prepare_data()
  # libri_data_module.setup()
  
  # ---------------------
  # model
  # ---------------------

  model = ParallelSpeechAndTextModel()
  
  # ---------------------
  # trainer
  # ---------------------

  accelerator = "gpu" if torch.cuda.is_available() else "cpu"
  strategy = "ddp" if torch.cuda.device_count > 1 else None
  trainer = Trainer(logger=wandb_logger, callbacks=[libri_logging_callback], accelerator=accelerator, strategy=strategy)
  
  # ---------------------
  # training
  # ---------------------

  wandb_logger.watch(model)
  trainer.fit(model=model, datamodule=libri_data_module)
  
  # ---------------------
  # testing
  # ---------------------

  trainer.test(model=model, datamodule=libri_data_module)
  wandb_logger.unwatch(model)


if __name__ == "__main__":
    main()