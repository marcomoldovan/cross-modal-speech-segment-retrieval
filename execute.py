import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from models import ParallelSpeechAndTextModel
from datamodules import LibriSpeechDataModule
from callbacks import LibriLoggingCallback, LitProgressBar
from utils import construct_arguments_parser, build_config_from_args, build_run_name_from_config 



def main():
  
  # ---------------------
  # args
  # ---------------------
  
  parser = construct_arguments_parser()
  config = build_config_from_args(parser)
  run_name = build_run_name_from_config(config)
  
  # ---------------------
  # non-essential
  # ---------------------
  
  #TODO log artifacts and store them on GDrive or wandb cloud
  
  wandb_logger = WandbLogger(name=run_name, project="cross-modal-speech-segment-retrieval", log_model='all')
  libri_logging_callback = LibriLoggingCallback()
  progress_bar = LitProgressBar()
  
  # ---------------------
  # data
  # ---------------------

  libri_data_module = LibriSpeechDataModule(config)
  
  # ---------------------
  # model
  # ---------------------

  model = ParallelSpeechAndTextModel(config)
   
  # ---------------------
  # trainer
  # ---------------------

  #TODO accumulate_grad_batches
  trainer = Trainer(logger=wandb_logger, callbacks=[libri_logging_callback, progress_bar], accelerator=config.accelerator, gpus=config.num_gpus, strategy=config.strategy)
  
  # ---------------------
  # training
  # ---------------------

  trainer.fit(model=model, datamodule=libri_data_module)
  
  # ---------------------
  # testing
  # ---------------------

  trainer.test(model=model, datamodule=libri_data_module)


if __name__ == "__main__":
    main()