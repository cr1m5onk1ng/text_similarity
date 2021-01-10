import argparse
import torch
from torch import nn 
from torch.nn import functional as F
import transformers
import numpy as np
from src.dataset.entailment_dataset import *
from src.configurations import config as config
from src.utils.metrics import AccuracyMeter, F1Meter
from src.models.sentence_encoder import SiameseSentenceEmbedder
from src.models.losses import SoftmaxLoss
from src.modules.pooling import *
from src.training.train import Trainer
from src.training.learner import Learner
from src.utils.utils import load_file, save_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs")
    parser.add_argument('--name', type=str, dest="config_name")
    parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--train_path', dest="train_path", type=str, default="../data/xnli/train-en.tsv")
    parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/xnli/test-en.tsv")
    parser.add_argument('--save_path', dest="save_path", type=str, default="../models/trained_models")
    parser.add_argument('--freeze', dest="freeze_weights", type=bool, default=False)
    parser.add_argument('--pretrained', dest="use_pretrained_embeddings", type=bool, default=False)
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768*3)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bert-base-cased")
    parser.add_argument('--setype', type=str, dest="sense_embeddings_type", default="ares_multi")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
    parser.add_argument('--sense_features', type=bool, dest="senses_as_features", default=True)
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")
    

    POOLING_STRATEGIES = {
        "avg": AvgPoolingStrategy,
        "cls": CLSPoolingStrategy,
        "sense": SiameseSensePoolingStrategy
    }

    args = parser.parse_args()

    # Training on a combination of mnli and snli

    #train_dataset = EntailmentDataset.build_dataset([args.train_path])

    #valid_dataset = EntailmentDataset.build_dataset([args.valid_path])

    train_data_loader = load_file("../dataset/cached/train_xnli_en_16")#SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="standard")
    valid_data_loader = load_file("../dataset/cached/valid_xnli_en_16")#SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="standard")

    #save_file(train_data_loader, "../dataset/cached/", "train_xnli_en_16")
    #save_file(valid_data_loader, "../dataset/cached/", "valid_xnli_en_16")


    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}


    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = 3,
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

    model = SiameseSentenceEmbedder(
        params = configuration,
        loss = SoftmaxLoss,
        pooling_strategy = POOLING_STRATEGIES[args.pooling_strategy],
        pooler = EmbeddingsPooler,
        merge_strategy = SentenceBertCombineStrategy
    )

    num_train_steps = len(train_data_loader) * args.epochs
    num_warmup_steps = int(num_train_steps*0.1)

    learner = Learner(
        config_name=args.config_name, 
        model=model, 
        lr=args.lr, 
        bs=args.batch_size, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        device=args.device, 
        fp16=args.mixed_precision, 
        metrics=metrics
    )

    trainer = Trainer(
        args.config_name, 
        train_data_loader, 
        valid_data_loader, 
        args.epochs, 
        configuration=learner, 
        direction=args.direction, 
        measure=args.measure
    )

    trainer.execute(write_results=True)





