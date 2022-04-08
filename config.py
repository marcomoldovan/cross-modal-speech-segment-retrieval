"""
  Serves as a base configuration file for the model.
  Individual arguments can be overwritten by the user during training start.
  Run names are built from the arguments.
"""

import torch
from transformers import Wav2Vec2Config, BertConfig, PretrainedConfig


class ParallelSpeechAndTextModelConfig(PretrainedConfig):
  def __init__(self, 
               model_name='PSTM',
               bert_pretrained_name='bert-base-uncased', 
               wav2vec2_pretrained_name='facebook/wav2vec2-base-960h',
               speech_output_pooling_strategy='mean',
               pretraining_contrastive_loss_fn='TripletMarginLoss',
               train_last_n_layers=0,
               training_mode='pretrain',
               train_batch_size=64,
               val_batch_size=512,
               num_epochs=10,
               accumulate_grad_batches=1,
               dataset_name='librispeech'):
    
    # model related
    self.model_name = model_name
    self.bert_pretrained_name = bert_pretrained_name
    self.wav2vec2_pretrained_name = wav2vec2_pretrained_name
    self.speech_output_pooling_strategy = speech_output_pooling_strategy
    self.pretraining_contrastive_loss_fn = pretraining_contrastive_loss_fn
    
    # training related
    self.train_last_n_layers = train_last_n_layers
    self.training_mode = training_mode
    self.train_batch_size = train_batch_size
    self.val_batch_size = val_batch_size
    self.num_epochs = num_epochs
    self.accumulate_grad_batches = accumulate_grad_batches
    
    # data related
    self.dataset_name = dataset_name
    
    # hardware related
    self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    self.num_gpus = torch.cuda.device_count()
    self.strategy = "ddp" if self.num_gpus > 1 else None
    
    

class CrossModalLanguageModelConfig(PretrainedConfig):
  pass
