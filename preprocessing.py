from typing import Dict, List, Union
import torch
import soundfile as sf
from transformers import BertTokenizerFast, Wav2Vec2FeatureExtractor

class LibriPreprocessor:
  def __init__(self):
    self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    self.extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
  
  
  def speech_file_to_array_fn(self, data):
    speech_array, sampling_rate = sf.read(data["file"])
    data["speech"] = speech_array
    data["sampling_rate"] = sampling_rate
    data["target_text"] = data["text"]
    return data
    
    
  def prepare_dataset(self, data):    
    # check that all files have the correct sampling rate
    assert (
        len(set(data["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {self.extractor.feature_extractor.sampling_rate}."

    data["input_values"] = self.extractor(data["speech"], sampling_rate=data["sampling_rate"][0]).input_values
    
    tokenized_batch = self.tokenizer(data["target_text"], padding='longest', max_length=128, pad_to_max_length=False)
    data['input_ids'] = tokenized_batch['input_ids']
    data['attention_mask_text'] = tokenized_batch['attention_mask']
    data['token_type_ids_text'] = tokenized_batch['token_type_ids']
    
    return data


  def __call__(
    self,
    batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
    """
    Collate function to be used when training with PyTorch Lightning.
    Args:
        extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        tokenizer (:class:`~transformers.BertTokenizerFast`)
            The tokenizer used for proccessing the data.
        features (:obj:`List[Dict[str, Union[List[int], torch.Tensor]]]`):
            A list of features to be collated.
    Returns:
        :obj:`Dict[str, torch.Tensor]`: A dictionary of tensors containing the collated features.
    """ 
    input_features = [{"input_values": feature["input_values"]} for feature in batch]
    input_sentences = [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask_text"]} for feature in batch]
    
    speech_batch = self.extractor.pad(
        input_features,
        padding='longest',
        return_tensors="pt",
        )
    text_batch = self.tokenizer.pad(
        input_sentences,
        padding='longest',
        return_tensors='pt'
    )
    
    return speech_batch, text_batch


def collate_fn_spotify(batch):
  pass