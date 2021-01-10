from src.utils import utils
from src.configurations import classifier_config as config
from src.configurations import embeddings_config as embeddings_config
from src.dataset.dataset import *
from src.modules.contextual_embedder import ContextualEmbedder
from src.evaluation.evaluators import ParaphraseEvaluator
from src.models.modeling import SiameseSentenceEmbedder, MBERTClassifier
from src.utils.metrics import SimilarityAccuracyMeter, SimilarityAveragePrecisionMeter, SimilarityF1Meter, AccuracyMeter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = {
    "mbert_sense": "training_mbert_paws_sense_avg_features_5epochs",
    "siamese_online_contrastive": "training_siamese_paws_sense_online_contrastive_combine_5epochs",
    "siamese_oc_cls": "training_siamese_paws_sense_online_contrastive_cls_pooling_5epochs",
    "no_sense": "training_siamese_paws_no_sense_online_contrastive_8epochs",
    "no_sense_softmax": "training_siamese_paws_no_sense_softmax_5epochs"
}
MODEL_NAME = MODEL_NAMES["mbert_sense"]
PRETRAINED_PATH = f"../training/trained_models/{MODEL_NAME}"

valid_data_loader = utils.load_file(f"../dataset/cached/pawsx_test_all_languages_16")
    
#metrics = {"validation": [SimilarityAveragePrecisionMeter, SimilarityAccuracyMeter]}
metrics = {"validation": [AccuracyMeter]}

"""
model = SiameseSentenceEmbedder(
    train_model = False,
    use_sense_embeddings=False
)
"""

model = MBERTClassifier(
    train_model = False,
    use_sense_embeddings=True,
    senses_as_features=True
)


model.load_pretrained(PRETRAINED_PATH)

#model_2.load_pretrained(PRETRAINED_PATH_2)

evaluator = ParaphraseEvaluator(
    model = model,
    data_loader = valid_data_loader,
    device = config.DEVICE,
    metrics = metrics,
    fp16=True,
    verbose=True
)


evaluator.evaluate()







