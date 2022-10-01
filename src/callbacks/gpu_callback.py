"""Monitor GPU usage and memory usage"""
import gc
import math
import torch
import pytorch_lightning as pl
from typing import Any, Optional
from torch.optim import Optimizer
from pytorch_lightning import Callback
from pytorch_lightning.core.hooks import DataHooks, ModelHooks

from src.utils import print_gpu_usage
        
        
def print_current_hook(hook):
    def wrapper():
        print(f'################: {hook.__name__} ##################')
        hook()
        print('#####################################################')
    return wrapper


class GPUMonitoringCallback(Callback, DataHooks, ModelHooks):
    
    def __init__(self, print_full_trace=False) -> None:
        super().__init__()
        self.print_full_trace = print_full_trace
        
    
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        print('############# on_before_batch_transfer ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        print('############# on_after_batch_transfer ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_train_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        print('############# on_before_backward ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_after_backward ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
     
    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer, opt_idx: int) -> None:
        print('############# on_before_optimizer_step ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
      
    def on_before_zero_grad(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer) -> None:
        print('############# on_before_zero_grad ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
      
    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_sanity_check_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_sanity_check_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
      
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_train_epoch_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_train_epoch_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_validation_epoch_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_validation_epoch_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_test_epoch_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('############# on_test_epoch_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
     
    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        print('############# on_validation_batch_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
     
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        print('############# on_validation_batch_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
     
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, unused: int = 0) -> None:
        print('############# on_train_batch_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
      
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, unused: int = 0) -> None:
        print('############# on_train_batch_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        print('############# on_test_batch_start ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
    
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        print('############# on_test_batch_end ##############')
        print_gpu_usage(self.print_full_trace)
        print('#####################################################')
        
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        print('############# on_exception ##############')
        print_gpu_usage()
        print('#####################################################')
        