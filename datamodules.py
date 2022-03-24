from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, BertTokenizerFast
from pytorch_lightning import LightningDataModule
from preprocessing import LibriPreprocessor, collate_fn_spotify


##############################
######## LibriSpeech #########
##############################

class LibriSpeechDataModule(LightningDataModule):
  def __init__(self):
    super().__init__()
    self.preprocessor = LibriPreprocessor()
    
    
  def prepare_data(self):
    load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')#'librispeech_asr', 'clean')
        
    
  def setup(self, stage=None):
    
    # Assign train/val datasets for use in dataloaders
    if stage == "fit" or stage is None:
      libri = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')#'librispeech_asr', 'clean')
      libri = libri.map(self.preprocessor.speech_file_to_array_fn, remove_columns=['chapter_id', 'id', 'speaker_id'])
      libri = libri.map(self.preprocessor.prepare_dataset, batch_size=8, num_proc=4, batched=True)
      libri_full = LibriSpeechDataset(libri)
      self.libri_train, self.libri_val = random_split(libri_full, [int(0.9 * len(libri_full)), len(libri_full) - int(0.9 * len(libri_full))])
    
    # Assign test dataset for use in dataloader(s)
    if stage == "test" or stage is None:
      self.libri_test = LibriSpeechDataset()
    
    if stage == "predict" or stage is None:
      pass
  
  
  def train_dataloader(self):
    return DataLoader(self.libri_train, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=self.preprocessor)
    
    
  def val_dataloader(self):
    return DataLoader(self.libri_val, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=self.preprocessor)
    
    
  def test_dataloader(self):
    return DataLoader(self.libri_test, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=self.preprocessor)
  
  
  
  
class LibriSpeechDataset(Dataset):
  def __init__(self, libri_dataset):
    self.libri_dataset = libri_dataset
  
  
  def __len__(self):
    return len(self.libri_dataset)
  
  
  def __getitem__(self, index):
    return self.libri_dataset[index]
    


##################################
######## SpotifyPodcasts #########
##################################


class SpotifyPodcastsDataModule(LightningDataModule):
  def __init__(self):
    super().__init__()
    self.extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
      
      
  def prepare_data(self):
    pass  
    
    
  def setup(self, stage=None):
    if stage == "fit" or stage is None:
      spotify_full = SpotifyDataset(self.extractor, self.tokenizer, split='train')
      self.spotify_train, self.spotify_val = random_split(spotify_full, [int(0.9 * len(spotify_full)), len(spotify_full) - int(0.9 * len(spotify_full))])
    
    # Assign test dataset for use in dataloader(s)
    if stage == "test" or stage is None:
      self.spotify_test = SpotifyDataset(self.extractor, self.tokenizer, split='train')
    
    if stage == "predict" or stage is None:
      pass
  
  
  def train_dataloader(self):
    return DataLoader(self.spotify_train, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn_spotify)
    
    
  def val_dataloader(self):
    return DataLoader(self.spotify_val, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn_spotify)
    
    
  def test_dataloader(self):
    return DataLoader(self.spotify_test, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn_spotify)
    
    

class SpotifyDataset(Dataset):
  def __init__(self, feature_extractor, tokenizer):
    self.spotify_dataset = load_dataset('spotify_podcasts', 'clean')
    self.extractor = feature_extractor
    self.tokenizer = tokenizer
    
  
  def __len__(self):
    return len(self.spotify_dataset)
  
  
  def __getitem__(self, index):
    return self.spotify_dataset[index]
  
