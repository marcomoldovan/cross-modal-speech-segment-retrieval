from distutils.command.config import config
import torch
import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerEncoder
from transformers import BertModel, Conv1D, Wav2Vec2Model
from torchmetrics import RetrievalNormalizedDCG
from torch.nn import TripletMarginWithDistanceLoss, Linear

class ParallelSpeechAndTextModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.speech_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    self.text_model = BertModel.from_pretrained('bert-base-uncased')
    self._freeze_network_layers(self.speech_model, self.text_model)
    
    # self.text_pooling = Linear(self.text_model.hidden_size, self.text_model.pooling_size)
    # self.speech_pooling = Linear(self.speech_model.hidden_size, self.speech_model.pooling_size)
    
    #TODO set margin and distance function, possibly from config
    self.triplet_loss = TripletMarginWithDistanceLoss()
    self.ndcg = RetrievalNormalizedDCG(top_k=10)
    
    # TODO is this redundant to the on_fit_start callback?
    self.save_hyperparameters()
    
    
  def forward(self, speech_input, text_input):
    speech_output = self.speech_model(speech_input)
    text_output = self.text_model(text_input)
    return text_output, speech_output
  
  
  def training_step(self, batch, batch_idx):
    speech_input = batch['input_values']
    text_input = batch['input_ids']
    
    text_anchors, speech_positives = self(speech_input, text_input)
    speech_negatives = self._sample_negatives(speech_positives)
    loss = self.triplet_loss(text_anchors, speech_positives, speech_negatives)
    
    return loss
  
  
  def validation_step(self, batch, batch_idx):
    speech_input = batch['input_values']
    text_input = batch['input_ids']
    
    text_output, speech_output = self(speech_input, text_input)
    #TODO sample text queries for validation
    ndcg_score = self.ndcg(text_output, speech_output)
    return ndcg_score
  
  
  def test_step(self, batch, batch_idx):
    language_input, speech_input = batch
    text_output, speech_output = self(language_input, speech_input)
    loss = torch.mean(torch.abs(text_output - speech_output))
    return {'test_loss': loss}
  
  
  def configure_optimizers(self):
    #TODO add scheduler to optimizer config or return list of optimizers and schedulers
    return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    
  def load_from_checkpoint(self, checkpoint_path: str) -> None:
    #TODO is this redundant? does the default checkpoint loading work?
    return super().load_from_checkpoint(checkpoint_path)
  
  
  def _sample_negatives(self, speech_outputs):
    #TODO sample negatives from the speech model
    pass
  
  
  def _sample_validation_query(self):
    #TODO sample validation query
    pass
  
  
  def _freeze_network_layers(self, speech_model, text_model, compute_grads_on_last_x_layers=1):
    #TODO freeze network layers
    pass  
  
  
  
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
  
