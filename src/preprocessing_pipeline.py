import hydra
from omegaconf import DictConfig

from src.preprocessing.preencode_libri import LibriPreprocessor
from src import utils

log = utils.get_logger(__name__)


def preprocess(config: DictConfig) -> None:
    
    # 0. instantiate preprocessor
    log.info(f"Instantiating preprocessor <{config.preprocessing._target_}>")
    preprocessor: LibriPreprocessor = hydra.utils.instantiate(config.preprocessing)

    # 1. Load data
    log.info(f"Loading dataset <{config.preprocessing.dataset_name}/{config.dataset_split}>")
    preprocessor.load_dataset(config.dataset_split)
    
    # 2. Speech file to array
    log.info("Copying speech array and sampling rate.")
    preprocessor.speech_file_to_array()
    
    # 3. Filter audio samples longer than max_audio_length
    log.info(f"Filtering audio samples longer than {config.max_audio_length} seconds.")
    preprocessor.filter_long_audio(config.max_audio_length)
    
    # 4. extract features from audio
    log.info(f"Extracting features from audio and tokenizing text.")
    preprocessor.extract_features_and_tokenize()
    
    # 5. encode text
    log.info(f"Encoding text with BERT: <{config.preprocessing.text_model_name}>.")
    preprocessor.encode_text()
    
    # 6. save data
    save_loc_dict = {
        'workdir': config.path_local,
        'HD': config.path_hd,
        'drive': config.path_colab,
    }
    log.info(f"Saving data to <{save_loc_dict[config.save_in]}{config.save_path}>.")
    preprocessor.save_dataset(save_loc_dict[config.save_in], config.save_path)
    
    