import torch
import numpy as np

from typing import Dict, List, Union
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, Wav2Vec2FeatureExtractor



class SpokenSQuADDataset(Dataset):
    def __init__(self, libri_dataset):
        self.libri_dataset = libri_dataset
    
    
    def __len__(self):
        return len(self.libri_dataset)
    
    
    def __getitem__(self, index):
        return self.libri_dataset[index]
    
    
class SpokenSQuADEmbeddedDataset(Dataset):
    def __init__(self, libri_dataset):
        self.libri_dataset = libri_dataset
    
    
    def __len__(self):
        return len(self.libri_dataset)
    
    
    def __getitem__(self, index):
        return self.libri_dataset[index]
    
    

class SpokenSQuADCollator:
    def __inti__(
        self,
        load_preprocessed_data=False,
        load_encoded_text=False,
        pretrained_speech_model="ntu-spml/distilhubert",
        speech_max_length=80000,
        pretrained_text_model="google/bert_uncased_L-2_H-768_A-12",
        text_max_length=32,
    ):
        self.load_preprocessed_data = load_preprocessed_data
        self.load_encoded_text = load_encoded_text
        self.speech_max_length = speech_max_length
        self.text_max_length = text_max_length
        
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_speech_model)
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_text_model)
        
        
    def collate_fn_for_latent_features_and_text_embeddings(
        self,
        batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    
    def collate_fn_for_latent_features(
        self,
        batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    
    def collate_fn_for_input_values(
        self,
        batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    
    def __call__(self, batch):
        if self.load_preprocessed_data and self.load_encoded_text:
            speech_batch, text_batch = self.collate_fn_for_latent_features_and_text_embeddings(batch)
        elif self.load_preprocessed_data and not self.load_encoded_text:
            speech_batch, text_batch = self.collate_fn_for_latent_features(batch)
        else:
            speech_batch, text_batch = self.collate_fn_for_input_values(batch)
            
        return speech_batch, text_batch