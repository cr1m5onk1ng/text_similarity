import argparse
from src.dataset.sts_dataset import StsDataset
import torch
from torch import nn 
from torch.nn import functional as F
import transformers
import numpy as np
from src.dataset.entailment_dataset import *
from src.configurations import config as config
from src.utils.metrics import AccuracyMeter, EmbeddingSimilarityMeter, F1Meter
from src.models.sentence_encoder import SiameseSentenceEmbedder
from src.models.losses import SoftmaxLoss
from src.modules.pooling import *
from src.training.train import Trainer
from src.training.learner import Learner
from src.utils.utils import load_file, save_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=4)
    parser.add_argument('--name', type=str, dest="config_name")
    parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--train_path', dest="train_path", type=str, default="../data/sts/stsbenchmark.tsv")
    parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/sts/stsbenchmark.tsv")
    parser.add_argument('--save_path', dest="save_path", type=str, default="./trained_models")
    parser.add_argument('--pretrained_path', dest="pretrained_path", type=str, default="../training/trained_models/sencoder-bert-base-nli")
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bert-base-uncased")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")
    

    POOLING_STRATEGIES = {
        "avg": AvgPoolingStrategy,
        "cls": CLSPoolingStrategy,
        "sense": SiameseSensePoolingStrategy
    }

    args = parser.parse_args()

    metrics = {"training": [EmbeddingSimilarityMeter], "validation": [EmbeddingSimilarityMeter]}


    model_config = config.ModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        freeze_weights = False,
        context_layers = (-1,)
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model),
      
    )

    train_dataset = StsDataset.build_dataset(args.train_path, mode="train")
    valid_dataset = StsDataset.build_dataset(args.valid_path, mode="dev")
    print("Building batches. This may take a while.")
    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="standard", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="standard", config=configuration)
    print("Done.")
    #save_file(train_data_loader, "../dataset/cached/sts", "train_sts_16")
    #save_file(valid_data_loader, "../dataset/cached/sts", "valid_sts_16")

    embedder_config = transformers.AutoConfig.from_pretrained(configuration.model)
    embedder = transformers.AutoModel.from_pretrained(configuration.model, config=embedder_config)

    model = SiameseSentenceEmbedder(
        params = configuration,
        context_embedder=embedder,
        loss = CosineSimilarityLoss,
        pooling_strategy = POOLING_STRATEGIES[args.pooling_strategy],
        pooler = EmbeddingsPooler,
        merge_strategy = EmbeddingsSimilarityCombineStrategy
    )

    model.load_pretrained(args.pretrained_path)

    num_train_steps = len(train_data_loader) * args.epochs
    num_warmup_steps = int(num_train_steps*0.1)

    learner = Learner(
        params = configuration,
        config_name=args.config_name, 
        model=model, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
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





