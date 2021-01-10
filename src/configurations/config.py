import pickle
import torch
import transformers
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union

@dataclass
class ModelParameters:
    model_name: str
    hidden_size: int
    num_classes: int
    use_pretrained_embeddings: bool = False
    freeze_weights: bool = True
    context_layers: Tuple[int] = (-1,)


@dataclass
class SenseModelParameters(ModelParameters):
    senses_as_features: bool = True


@dataclass
class Configuration:
    model_parameters: Union[ModelParameters, None]
    model: str
    save_path: str
    sequence_max_len: int
    dropout_prob: float
    lr: float
    batch_size: int
    epochs: int
    device: torch.device
    embedding_map: Dict[str, torch.Tensor]
    bnids_map: Dict[str, torch.Tensor]
    tokenizer: transformers.AutoTokenizer
    pretrained_embeddings_dim: int
    senses_as_features: bool


def load_file(path):
    file_name = open(path, 'rb')
    d = pickle.load(file_name)
    file_name.close()
    return d

EMBEDDINGS_PATHS = {
    "ares_multi": "../embeddings/ares_embed_map",
    "ares_mono": "../embeddings/ares_embed_map_large",
    "lmms": "../embeddings/lmms_embed_map"
}

MODELS = {
    "base": "bert-base-cased",
    "large": "bert-large-cased",
    "multi": "bert-base-multilingual-cased",
    "xlm-r": "xlm-roberta-base",
    "roberta": "roberta-base",
    "roberta-l": "roberta-large"
}

LEARNING_RATES = {
    "pawsx": 2e-5,
    "nli": 2e-5,
    "wic": 4.456712240296184e-5,
}

DIMENSIONS_MAP = {
    "ares_multi": 768*2,
    "ares_mono": 1024*2,
    "lmms": 1024
}


CONFIG = Configuration(
    model_parameters = None,
    model = MODELS["large"],
    save_path = "../models",
    sequence_max_len = 256,
    dropout_prob = 0.1,
    lr = LEARNING_RATES["wic"],
    batch_size = 16,
    epochs = 3,
    device = torch.device("cuda"),
    embedding_map = load_file(EMBEDDINGS_PATHS["ares_mono"]),
    bnids_map = load_file("../data/bnids_map"),
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODELS["large"]),
    pretrained_embeddings_dim = DIMENSIONS_MAP["ares_mono"],
    senses_as_features = True
)



