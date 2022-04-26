import sys
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ProgressBarBase, BaseFinetuning
import wandb


class LoggingCallback(Callback):
  
  def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    """Logs gradients, parameters and model topology."""
    trainer.logger.watch(pl_module)
    trainer.logger.log_hyperparams(pl_module.parameters())
  
  
  def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
    """Logs the loss and accuracy of the training step."""
    trainer.logger.log_metrics(outputs)
  
  
  def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    """Logs the MRR@val_batch_size of the validation step."""
    #TODO log table of given text, predicted speech and ground truth speech
    trainer.logger.log_metrics(outputs)
    
    
  def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    """Save model checkpoint at the end of each epoch."""
    pass
  
  
  def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    config = pl_module.config
    model_artifact = wandb.Artifact(name=config.model_name, description=config.run_name, type='model', metadata=config)
  
  
  def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    """Logs the MRR@test_batch_size of the test step."""
    pass
  
  
  def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    trainer.logger.finalize()
    trainer.logger.unwatch(pl_module)
    
    
    
class ToggleTrainableParamsPSTM(BaseFinetuning):
  def __init__(self, config):
    pass
    
    

class LitProgressBar(ProgressBarBase):

  def __init__(self):
    super().__init__()
    self.enable = True


  def disable(self):
    self.enable = False


  def on_train_batch_end(self, trainer, pl_module, outputs, batch_idx):
    super().on_train_batch_end(trainer, pl_module, outputs, batch_idx)
    percent = (self.train_batch_idx / self.total_train_batches) * 100
    sys.stdout.flush()
    sys.stdout.write(f'{percent:.01f} percent complete \r')
    
    
#TODO add callback BaseFinetunig
