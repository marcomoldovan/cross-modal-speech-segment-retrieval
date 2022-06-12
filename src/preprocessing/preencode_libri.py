import os
import torch
from transformers import BertTokenizerFast, BertModel, Wav2Vec2FeatureExtractor
from datasets.load import load_dataset

from src.models.components.encoder import HubertConvFeatureExtractorWrapper

class LibriPreprocessor:
    def __init__(
        self, 
        dataset_name: str = 'librispeech_asr',
        text_model_name: str = 'google/bert_uncased_L-2_H-768_A-12',
    ):
        assert torch.cuda.is_available(), "CUDA is not available, should run on GPU"
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('ntu-spml/distilhubert')
        self.feature_encoder = HubertConvFeatureExtractorWrapper.from_pretrained('ntu-spml/distilhubert')
        self.feature_encoder.eval()
        
        self.tokenizer = BertTokenizerFast.from_pretrained(text_model_name)
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.text_model.eval()
        
        self.dataset_name = dataset_name
                
        self.dataset = None
        
        
    def load_dataset(self, dataset_split: str = 'train.360'):
        self.dataset = load_dataset(self.dataset_name, 'clean', split=dataset_split)
        
        
    def _speech_file_to_array(self, data):
        data['speech'] = data['audio']['array']
        data['sampling_rate'] = data['audio']['sampling_rate']
        return data
    
    
    def speech_file_to_array(self):
        self.dataset = self.dataset.map(
            self._speech_file_to_array, 
            remove_columns=['file', 'audio', 'speaker_id', 'chapter_id', 'id']
        )
    
    
    def filter_long_audio(self, max_audio_length: int = 16):
        self.dataset = self.dataset.filter(
            lambda x: len(x['speech'])//x['sampling_rate'] < max_audio_length, 
            num_proc=os.cpu_count()
        )
        
        
    def _extract_features_and_tokenize(self, data):
        # check that all files have the correct sampling rate
        assert (
            len(set(data['sampling_rate'])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self.feature_extractor.sampling_rate}."
        
        # extract and pad input values
        input_values = self.feature_extractor(data['speech'], sampling_rate=data['sampling_rate'][0])
        data['input_values'] = input_values.input_values
        padded_input_values = self.feature_extractor.pad(input_values, return_tensors='pt')
        
        # compute the latent features from the conv module
        import torch
        with torch.no_grad():
            input_values = padded_input_values['input_values'].to(self.device)
            latent_features = self.feature_encoder(input_values).transpose(1, 2)
            latent_features = latent_features.cpu().numpy()
            data['latent_features'] = latent_features
            
        # tokenize text
        tokenized_batch = self.tokenizer(data['text'], padding='longest', max_length=128, pad_to_max_length=False)
        data['input_ids'] = tokenized_batch['input_ids']
        data['attention_mask_text'] = tokenized_batch['attention_mask']
        data['token_type_ids_text'] = tokenized_batch['token_type_ids']
            
        return data
            
        
    def extract_features_and_tokenize(self):
        self.feature_encoder.cuda()
        self.dataset = self.dataset.map(
            self._extract_features_and_tokenize, 
            batch_size=16, 
            num_proc=1, 
            batched=True, 
            remove_columns=['text', 'sampling_rate']
        )
        self.feature_encoder.cpu()
    
    
    def _encode_text(self, data):
        import torch
        with torch.no_grad():
            data['sentence_embedding'] = self.text_model(
                input_ids=data['input_ids'], 
                attention_mask=data['attention_mask_text'], 
                token_type_ids=data['token_type_ids_text']
            ).pooler_output.cpu()
    
    
    def encode_text(self):
        self.text_model.cuda()
        self.dataset = self.dataset.map(
            self._encode_text, 
            batch_size=16, 
            num_proc=1, 
            batched=True, 
            remove_columns=['input_ids', 'attention_mask_text', 'token_type_ids_text']
        )
        self.text_model.cpu()
    
    
    def save_dataset(
        self, 
        save_in: str,
        save_path: str,
    ):
        self.dataset.save_to_disk(f'{save_in}/{save_path}')
