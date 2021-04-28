from src.utils import utils
from src.configurations import classifier_config as config
from src.configurations import embeddings_config as embeddings_config
from src.dataset.dataset import *
from src.evaluation.evaluators import RetrievalEvaluator
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.utils.metrics import RetrievalAccuracyMeter, cos_sim
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = {
    "mbert_avg": "mbert_sense_features_avg",
    "mbert_cls": "mbert_sense_cls_features",
    "baseline": "mbert_no_sense_cls",
    "siamese_contrastive_cls": "training_siamese_paws_sense_online_contrastive_cls_pooling_5epochs"
}

MODEL_NAME = MODEL_NAMES["baseline"]
PRETRAINED_PATH = f"../training/trained_models/{MODEL_NAME}"
    
metrics = {"validation": [RetrievalAccuracyMeter]}

model = MBERTClassifier(
    strategy="cls",
    train_model = False,
    use_sense_embeddings=False
)

model.load_pretrained(PRETRAINED_PATH)

model.to(config.DEVICE)

model.eval()


language_pairs = ['eng-jpn', 'eng-deu', 'eng-fra', 'eng-spa', 'eng-kor']
dev_paths = [f'../data/tatoeba/parallel-sentences/Tatoeba-{pair}-dev.tsv.gz' for pair in language_pairs]

langs = {
    "jp": 0,
    "de": 1,
    "fr": 2,
    "es": 3,
    "ko": 4
}

lang = langs["jp"]

valid_dataset = ParallelDataset.build_dataset([dev_paths[lang]])

valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="tatoeba")

metrics = {"validation": [RetrievalAccuracyMeter]}

print(f"Lang: {language_pairs[lang]}")

evaluator = RetrievalEvaluator(
    model = model,
    data_loader=valid_data_loader,
    device=config.DEVICE,
    metrics=metrics,
    fp16=True,
    verbose=True,
    return_predictions = False

)

evaluator.evaluate()

