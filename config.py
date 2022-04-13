"""
  Serves as a base configuration file for the model.
  Individual arguments can be overwritten by the user during training start.
  Run names are built from the arguments.
"""

import torch
from transformers import PretrainedConfig


class ParallelSpeechAndTextModelPretrainingConfig(PretrainedConfig):
  def __init__(self, 
               model_name='PSTM',
               hidden_size=768,
               bert_pretrained_name='bert-base-uncased', 
               wav2vec2_pretrained_name='facebook/wav2vec2-base-960h',
               speech_output_pooling_strategy='mean',
               pretraining_contrastive_loss_fn='TripletMarginLoss',
               train_last_n_layers=1,
               training_mode='pretrain',
               num_epochs=10,
               early_stopping_patience=5,
               accumulate_grad_batches=1,
               dataset_name='librispeech',
               train_batch_size=64,
               val_batch_size=512,
               test_batch_size=512,
               project_name='cross-modal-speech-segment-retrieval',
               project_entity=None,
               run_name=None):
    
    # model related
    self.model_name = model_name
    self.hidden_size = hidden_size
    self.bert_pretrained_name = bert_pretrained_name
    self.wav2vec2_pretrained_name = wav2vec2_pretrained_name
    self.speech_output_pooling_strategy = speech_output_pooling_strategy
    
    # training related
    self.pretraining_contrastive_loss_fn = pretraining_contrastive_loss_fn
    self.train_last_n_layers = train_last_n_layers
    self.training_mode = training_mode
    self.num_epochs = num_epochs
    self.early_stopping_patience = early_stopping_patience
    self.accumulate_grad_batches = accumulate_grad_batches # can be a dict with an accumulation strategy for each epoch: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.GradientAccumulationScheduler.html#pytorch_lightning.callbacks.GradientAccumulationScheduler
    
    # data related
    self.dataset_name = dataset_name
    self.train_batch_size = train_batch_size
    self.val_batch_size = val_batch_size
    self.test_batch_size = test_batch_size
    
    # strategy related
    self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    self.num_gpus = torch.cuda.device_count()
    self.strategy = "ddp" if self.num_gpus > 1 else None
    
    # logging related
    self.project_name = project_name
    self.project_entity = project_entity
    self.run_name = run_name
    
    
    
class ParallelSpeechAndTextModelFinetuningConfig(PretrainedConfig):
  def __init__(self):
    pass
    
    

class CrossModalLanguageModelConfig(PretrainedConfig):
  def __init__(self,
               model_name='CMLM'):
    
    # model related
    self.model_name=model_name
