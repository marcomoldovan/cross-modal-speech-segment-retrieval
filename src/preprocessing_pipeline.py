import hydra
from omegaconf import DictConfig

from src.preprocessing.preencode_libri import LibriPreprocessor
from src import utils

log = utils.get_logger(__name__)


def preprocess(config: DictConfig) -> None:
    
    # 0. instantiate preprocessor
    log.info(f"Instantiating preprocessor <{config.preprocessing._target_}>")
    preprocessor: LibriPreprocessor = hydra.utils.instantiate(config.datamodule)

    # 1. Load data
    preprocessor.load_dataset()
    
    # 2. extract features from audio
    preprocessor.extract_features()
    
    # 3. encode text
    preprocessor.encode_text()
    
    # 4. save data
    preprocessor.save_dataset(config.datamodule.dataset_path)
    
    