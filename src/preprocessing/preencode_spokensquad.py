import torch
from transformers import BertModel
from datasets.load import load_dataset

from src.models.components.encoder import HubertConvFeatureExtractorWrapper

class SpokenSquadPreprocessor:
    def __init__(
        self, 
        text_model_name: str,
        dataset_split: str = 'train.360',
        split_to_shards: bool = False,
    ):
        assert torch.cuda.is_available(), "CUDA is not available, should run on GPU"
                
        self.feature_extractor = HubertConvFeatureExtractorWrapper.from_pretrained('ntu-spml/distilhubert')
        self.text_model = BertModel.from_pretrained(text_model_name)
        
        self.dataset_split = dataset_split
        
        
    def load_dataset(self):
        self.dataset = load_dataset('librispeech_asr', 'clean', split=self.dataset_split)
        
    def _extract_features(self, audio_path):
        raise NotImplementedError
    
    def extract_features(self, audio_path):
        self.dataset = self.dataset.map(self._extract_features)
    
    def _encode_text(self, text):
        raise NotImplementedError
    
    def encode_text(self, text):
        self.dataset = self.dataset.map(self._encode_text)
        
    def save_shards(self, dataset_path):
        raise NotImplementedError
    
    def save_dataset(self, dataset_path):
        raise NotImplementedError