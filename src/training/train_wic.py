import argparse
import torch
from torch import nn 
import transformers
from torch.nn import functional as F
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_curve, precision_recall_curve
from src.utils import utils as utils
from src.configurations import config as config
from src.models.losses import SoftmaxLoss
from src.modules.pooling import EmbeddingsCombineStrategy, WordSensePoolingStrategy, WordPoolingStrategy, EmbeddingsPooler
from src.models.word_encoder import WordClassifierModel
import src.dataset.wic_dataset as dataset
from src.utils.metrics import AccuracyMeter
from src.training.learner import Learner
from src.training.train import Trainer

if __name__ == "__main__":


    ##### -------- TRAINING PREPARATION --------- #####

    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs")
    parser.add_argument('--name', type=str, dest="config_name")
    parser.add_argument('--lr', type=float, dest="lr", default=4.5e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=8)
    parser.add_argument('--train_path', dest="train_path", type=str, default="../data/WiC/train/train.data.txt")
    parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/WiC/dev/dev.data.txt")
    parser.add_argument('--gold_train_path', dest="gold_train_path", type=str, default="../data/WiC/train/train.gold.txt")
    parser.add_argument('--gold_valid_path', dest="gold_valid_path", type=str, default="../data/WiC/dev/dev.gold.txt")
    parser.add_argument('--save_path', dest="save_path", type=str, default="../models/trained_models")
    parser.add_argument('--freeze', dest="freeze_weights", type=bool, default=True)
    parser.add_argument('--pretrained', dest="use_pretrained_embeddings", type=bool, default=True)
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=2048)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bert-large-cased")
    parser.add_argument('--setype', type=str, dest="sense_embeddings_type", default="ares_mono")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="standard")
    parser.add_argument('--sense_features', type=bool, dest="senses_as_features", default=True)
    

    POOLING_STRATEGIES = {
        "standard": WordPoolingStrategy,
        "sense": WordSensePoolingStrategy
    }

    args = parser.parse_args()


    processor = dataset.WicProcessor()
    train_dataset = processor.build_dataset(args.train_path, args.gold_train_path)
    valid_dataset = processor.build_dataset(args.valid_path, args.gold_valid_path)
    train_data_loader = dataset.WiCDataLoader.build_batches(train_dataset, args.batch_size)
    valid_data_loader = dataset.WiCDataLoader.build_batches(valid_dataset, args.batch_size)

    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = 2,
        use_pretrained_embeddings = args.use_pretrained_embeddings,
        freeze_weights = args.freeze_weights,
        context_layers = (-1, -2, -3, -4)
    )

    configuration = config.Configuration(
        model_parameters=model_config,
        model = args.model,
        save_path = args.save_path,
        sequence_max_len = args.seq_len,
        dropout_prob = args.dropout,
        lr = args.lr,
        batch_size = args.batch_size,
        epochs = args.epochs,
        device = torch.device(args.device),
        embedding_map = config.CONFIG.embedding_map,
        bnids_map = config.CONFIG.bnids_map,
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model),
        pretrained_embeddings_dim = config.DIMENSIONS_MAP[args.sense_embeddings_type],
        senses_as_features = args.senses_as_features
    )

    model = WordClassifierModel(
        params = configuration,
        loss = SoftmaxLoss,
        pooling_strategy = POOLING_STRATEGIES[args.pooling_strategy],
        pooler = EmbeddingsPooler
    )

    num_train_steps = len(train_data_loader) * args.epochs

    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]} 

    learner = Learner(
        config_name=args.config_name, 
        model=model, 
        lr=args.lr, 
        bs=args.batch_size, 
        steps=num_train_steps, 
        device=args.device, 
        fp16=args.mixed_precision, 
        metrics=metrics
    )

    trainer = Trainer(args.config_name, train_data_loader, valid_data_loader, args.epochs, configuration=learner)
    trainer.execute(write_results=True)