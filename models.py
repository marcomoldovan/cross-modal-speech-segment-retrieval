from distutils.command.config import config
import torch
import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerEncoder
from transformers import BertModel, Conv1D, Wav2Vec2Model
from sentence_transformers.util import semantic_search
from torch.nn import TripletMarginWithDistanceLoss, Linear, Tanh
from metrics import reciprocal_ranks, mean_reciprocal_rank
from losses import InfoNceLoss
from utils import count_parameters

class ParallelSpeechAndTextModel(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    
    # models and model heads
    self.speech_model = Wav2Vec2Model.from_pretrained(config.wav2vec2_pretrained_name)
    if config.speech_output_pooling_strategy == 'pooling_layer':
      self.speech_pooling_dense = Linear(self.config.hidden_size, self.config.hidden_size)
      self.speech_pooling_act = Tanh()
    self.text_model = BertModel.from_pretrained(config.bert_pretrained_name)
    self._freeze_network_layers_except_last_n(self.speech_model, self.text_model, config.train_last_n_layers)
    
    # loss function
    if config.pretraining_contrastive_loss_fn == 'TripletMarginLoss':
      #TODO set margin and distance function, possibly from config
      self.triplet_loss = TripletMarginWithDistanceLoss()  
    elif config.pretraining_contrastive_loss_fn == 'InfoNceLoss':
      self.info_nce_loss = InfoNceLoss()
        
    # TODO is this redundant to the on_fit_start callback?
    self.save_hyperparameters()
    
    print(count_parameters(self))
    
    
  def forward(self, speech_input, text_input):
    speech_output = self.speech_model(input_values=speech_input['input_values'])
    if self.config.speech_output_pooling_strategy == 'mean':
      speech_output['pooler_output'] = torch.mean(speech_output['last_hidden_state'], dim=1)
    elif self.config.speech_output_pooling_strategy == 'pooling_layer':
      speech_output['pooler_output'] = self.speech_pooling_act(self.speech_pooling_dense(speech_output['last_hidden_state']))
    text_output = self.text_model(input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])
    return text_output, speech_output
  
  
  def training_step(self, batch, batch_idx):
    speech_input = batch[0]
    text_input = batch[1]
    
    text_anchors, speech_positives = self(speech_input, text_input)
    speech_negatives = speech_positives['pooler_outputs'][torch.randperm(speech_positives['pooler_outputs'].shape[0]),:]
    loss = self.triplet_loss(text_anchors['pooler_outputs'], speech_positives['pooler_outputs'], speech_negatives)
    
    return {f'{self.config.pretraining_contrastive_loss_fn}': loss}
  
  
  def validation_step(self, batch, batch_idx):
    speech_input = batch[0]
    text_input = batch[1]
    
    text_output, speech_output = self(speech_input, text_input)
    pairwise_similarity_results = semantic_search(text_output['pooler_output'], speech_output['pooler_output'])
    rs = reciprocal_ranks(pairwise_similarity_results)
    mrr_score = mean_reciprocal_rank(rs)
    
    return {f'MRR@{self.config.val_batch_size}': mrr_score}
  
  
  def test_step(self, batch, batch_idx):
    speech_input = batch[0]
    text_input = batch[1]
    
    text_output, speech_output = self(speech_input, text_input)
    pairwise_similarity_results = semantic_search(text_output['pooler_output'], speech_output['pooler_output'])
    rs = reciprocal_ranks(pairwise_similarity_results)
    mrr_score = mean_reciprocal_rank(rs)
    
    return {f'MRR@{self.config.test_batch_size}': mrr_score}
  
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    return (
      {
        "optimizer": optimizer,
        "lr_scheduler": 
          {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "mrr_score"
          }
      }
    )
    
    
  def load_from_checkpoint(self, checkpoint_path: str) -> None:
    #TODO is this redundant? does the default checkpoint loading work?
    return super().load_from_checkpoint(checkpoint_path)
  
  
  def _freeze_network_layers_except_last_n(self, speech_model, text_model, train_last_n_layers):
    # freeze speech model layers
    for param in speech_model.feature_extractor.parameters():
      param.requires_grad = False
    for param in speech_model.feature_projection.parameters():
      param.requires_grad = False
    for param in speech_model.encoder.pos_conv_embed.parameters():
      param.requires_grad = False
    for param in speech_model.encoder.layer_norm.parameters():
      param.requires_grad = False
    for param in speech_model.encoder.dropout.parameters():
      param.requires_grad = False
    for i, encoder_layer in enumerate(speech_model.encoder.layers._modules):
      if i < (len(speech_model.encoder.layers._modules) - train_last_n_layers):
        for param in speech_model.encoder.layers[i].parameters():
          param.requires_grad = False
          
    # freeze text model layers
    for param in text_model.embeddings.parameters():
      param.requires_grad = False
    for i, encoder_layer in enumerate(text_model.encoder.layer._modules):
      if i < (len(text_model.encoder.layer) - train_last_n_layers):
        for param in text_model.encoder.layer[i].parameters():
          param.requires_grad = False
    
    
  
  
  
class CrossModalLanguageModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.conv_feature_extractpr = Conv1D()
    self.multimodal_encoder = TransformerEncoder()
    
    self.save_hyperparameters()
    
    
  def forward(self, speech_input, text_input):
    speech_output = self.speech_model(speech_input)
    text_output = self.language_model(text_input)
    return text_output, speech_output
  
  
  def training_step(self, batch, batch_idx):
    if config["contrastive_loss"] == "SimCLR":
      pass
    elif config["contrastive_loss"] == "TripletMarginWithDistance":
      pass
    else:
      raise ValueError("Invalid contrastive loss")
    
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'train_loss': loss}
  
  
  def validation_step(self, batch, batch_idx):
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'val_loss': loss}
  
  
  def test_step(self, batch, batch_idx):
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'test_loss': loss}
  
  
  def configure_optimizers(self):
    #TODO add scheduler to optimizer config or return list of optimizers and schedulers
    return torch.optim.Adam(self.parameters(), lr=1e-3)
  
  
  def save_hyperparameters():
    #TODO check whether this is necessary when using wandb
    pass
    
    
  def load_from_checkpoint(self, checkpoint_path: str) -> None:
    #TODO is this redundant? does the default checkpoint loading work?
    return super().load_from_checkpoint(checkpoint_path)
  
