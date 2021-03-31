import argparse

from transformers import BertForTokenClassification
from src.dataset.dataset import SmartParaphraseDataloader
import torch
from torch import nn 
import transformers
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn import metrics
from src.utils import utils as utils
from src.configurations import config as config
from src.models.losses import SoftmaxLoss
from src.modules.pooling import SequencePoolingStrategy, WordSensePoolingStrategy, WordPoolingStrategy
from src.models.word_encoder import WordEncoderModel
import src.dataset.wic_dataset as dataset
from src.utils.metrics import AccuracyMeter
from src.training.learner import Learner
from src.training.train import Trainer
from src.dataset.ner_dataset import NerDataset

LABELS_MAPPING = {
            "I-EVENT": 0,
            "I-TIME": 1,
            "I-time": 1,
            "B-time": 1,
            "B-ARTIFACT": 2,
            "B-PERCENT": 3,
            "I-PERCENT": 3,
            "I-ARTIFACT": 2,
            "I-OTHER": 5,
            "O": 6,
            "B-OTHER": 5,
            "I-DATE": 7,
            "I-ORGANIZATION": 8,
            "B-EVENT": 0,
            "B-event": 0,
            "I-event": 0,
            "B-MONEY": 9,
            "B-PERSON": 10,
            "B-person": 10,
            "I-NUMBER": 11,
            "I-MONEY": 9,
            "B-DATE": 7,
            "B-LOCATION": 12,
            "B-location": 12,
            "I-LOCATION": 12,
            "I-location": 12,
            "I-PERSON": 10,
            "B-ORGANIZATION": 8,
            "B-organization": 8,
            "I-organization": 8,
            "B-TIME": 1,
            "B-NUMBER": 11,
            "B-creative-work": 13,
            "I-creative-work": 13,
            "I-group": 14,
            "B-group": 14,
            "B-corporation": 8,
            "I-corporation": 8,
            "B-product": 15,
            "I-product": 15,
            "I-person": 10,
            "B-plant": 16,
            "I-plant": 16,
            "B-substance": 17,
            "I-substance": 17,
            "B-quantity": 18,
            "I-object": 2,
            "B-place": 12,
            "I-place": 12,
            "I-quantity": 18,
            "B-abstract": 19,
            "I-abstract": 19,
            "B-object": 2,
            "I-animal": 20,
            "B-animal": 20,
        }

if __name__ == "__main__":


    ##### -------- TRAINING PREPARATION --------- #####

    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=1)
    parser.add_argument('--name', type=str, dest="config_name")
    parser.add_argument('--lr', type=float, dest="lr", default=4.5e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--train_path', dest="train_path", type=str, default="../data/WiC/train/train.data.txt")
    parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/WiC/dev/dev.data.txt")
    parser.add_argument('--gold_train_path', dest="gold_train_path", type=str, default="../data/WiC/train/train.gold.txt")
    parser.add_argument('--gold_valid_path', dest="gold_valid_path", type=str, default="../data/WiC/dev/dev.gold.txt")
    parser.add_argument('--save_path', dest="save_path", type=str, default="../models/trained_models")
    parser.add_argument('--freeze', dest="freeze_weights", type=bool, default=False)
    parser.add_argument('--pretrained', dest="use_pretrained_embeddings", type=bool, default=False)
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bert-base-uncased")
    parser.add_argument('--setype', type=str, dest="sense_embeddings_type", default="ares_mono")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="standard")
    parser.add_argument('--sense_features', type=bool, dest="senses_as_features", default=False)
    

    POOLING_STRATEGIES = {
        "standard": SequencePoolingStrategy,
        "sense": WordSensePoolingStrategy
    }

    args = parser.parse_args()


    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = len(LABELS_MAPPING),
        use_pretrained_embeddings = args.use_pretrained_embeddings
    )

    configuration = config.WordModelConfiguration(
        model_parameters=model_config,
        model = args.model,
        save_path = args.save_path,
        sequence_max_len = args.seq_len,
        dropout_prob = args.dropout,
        lr = args.lr,
        batch_size = args.batch_size,
        epochs = args.epochs,
        device = torch.device(args.device),
        embedding_map = None,
        bnids_map = None,
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model),
        pretrained_embeddings_dim = None,
        senses_as_features = args.senses_as_features
    )

    train_dataset = NerDataset.from_conll("../data/NER/entity-recognition-datasets-master/data/WNUT17/CONLL-format/data/train/wnut17train.conll", return_tensors=True, config=configuration)
    valid_dataset = NerDataset.from_conll("../data/NER/entity-recognition-datasets-master/data/WNUT17/CONLL-format/data/dev/emerging.dev.conll", return_tensors=True, config=configuration)

    """
    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, args.batch_size, mode="word", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, args.batch_size, mode="word", config=configuration)
    """
    train_data_loader = DataLoader(train_dataset, args.batch_size)
    valid_data_loader = DataLoader(valid_dataset, args.batch_size)

    print(f"Train dataloader sample: {train_data_loader}")

    """
    model = WordEncoderModel(
        context_embedder = transformers.AutoModel.from_pretrained(args.model),
        params = configuration,
        loss = SoftmaxLoss,
        pooling_strategy = POOLING_STRATEGIES[args.pooling_strategy],
    )
    """
    auto_config = transformers.AutoConfig.from_pretrained(args.model)
    auto_config.num_labels = len(train_dataset.tag_to_labels) 
    model = BertForTokenClassification.from_pretrained(args.model, config=auto_config)

    num_train_steps = len(train_data_loader) * args.epochs

    #metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]} 

    learner = Learner(
        params=configuration,
        config_name=args.config_name, 
        model=model, 
        steps=num_train_steps, 
        fp16=args.mixed_precision, 
        metrics=None
    )

    trainer = Trainer(args.config_name, train_data_loader, valid_data_loader, args.epochs, configuration=learner, eval_in_train=True)
    trainer.execute(write_results=True)