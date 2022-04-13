import torch
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from models import ParallelSpeechAndTextModel
from datamodules import LibriSpeechDataModule
from callbacks import LoggingCallback, LitProgressBar
from setup import construct_arguments_parser, build_config_from_args, build_run_name_from_config, rebuild_config_object_from_wandb



#TODO add distributed training plugin from Ray: https://docs.ray.io/en/latest/ray-more-libs/ray-lightning.html
def main():
  
  # ---------------------
  # args
  # ---------------------
  
  parser = construct_arguments_parser()
  config = build_config_from_args(parser)
  run_name = build_run_name_from_config(config)
  
  with wandb.init(project=config.project_name, entity=config.project_entity, job_type="train", config=config) as run:
    
    run_config = run.config
    run_config_as_dict = run_config.as_dict()
    run_config_readable = rebuild_config_object_from_wandb(run_config_as_dict)
    run_config_readable.run_name = run_name

    wandb_logger = WandbLogger(experiment=run, name=run_name, log_model='all')

    # ---------------------
    # callbacks
    # ---------------------
    
    libri_logging_callback = LoggingCallback()
    progress_bar = LitProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval=None, log_momentum=True)
    early_stopping = EarlyStopping(monitor='mrr_score', min_delta=0.1, patience=run_config_readable.early_stopping_patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(filepath=run_config_readable.checkpoint_path, save_top_k=1, verbose=True, monitor='mrr_score', mode='max')
    
    # ---------------------
    # data
    # ---------------------

    libri_data_module = LibriSpeechDataModule(run_config_readable)
    
    # ---------------------
    # model
    # ---------------------
    
    model = ParallelSpeechAndTextModel(run_config_readable)
    
    # ---------------------
    # trainer
    # ---------------------

    trainer = Trainer(logger=wandb_logger, 
                      callbacks=[libri_logging_callback, progress_bar, early_stopping, lr_monitor, checkpoint_callback], 
                      accelerator=run_config_readable.accelerator, 
                      gpus=run_config_readable.num_gpus, 
                      strategy=run_config_readable.strategy, 
                      accumulate_grad_batches=run_config_readable.accumulate_grad_batches)
    
    # ---------------------
    # training
    # ---------------------

    trainer.fit(model=model, datamodule=libri_data_module)
    
    # ---------------------
    # testing
    # ---------------------

    trainer.test(model=model, datamodule=libri_data_module)
    
    # ---------------------
    # save artifact
    # ---------------------


if __name__ == "__main__":
    main()