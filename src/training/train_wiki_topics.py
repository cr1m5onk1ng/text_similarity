import argparse
from src.utils.wikipedia_extractor import WikipediaExtractor
from transformers import BertForSequenceClassification
from transformers import AutoConfig, AutoModel
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
from src.utils.metrics import AccuracyMeter
from src.training.learner import Learner
from src.training.train import Trainer
from src.dataset.wikipedia_dataset import WikipediaDataset, CATEGORIES
from src.dataset.dataset import KFoldStratifier
import random


if __name__ == "__main__":
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

    args = parser.parse_args()

    random.seed(43)

    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}

    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = len(CATEGORIES),
        use_pretrained_embeddings = args.use_pretrained_embeddings,
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

    config = AutoConfig.from_pretrained(args.model)

    config.num_labels = len(CATEGORIES)

    model = BertForSequenceClassification.from_pretrained(args.model, config=config)

    url = "https://ja.wikipedia.org/w/api.php?"
    lang = "ja"
    extractor = WikipediaExtractor(
        url = url,
        lang = lang
    )

    print("Searching page ids...")
    ids = extractor.extract_ids_from_categories(CATEGORIES, max_pages=50)
    print("Done.")
    print(f"Ids: {ids}")
    print()
    print("Building dataset.")
    files = list(utils.search_files("../data/wikipedia-dump/japanese/extracted/"))
    dataset = WikipediaDataset.from_collection(files, page_ids=ids, max_n_docs=1000)
    print("Done")

    dataset = WikipediaDataset.from_collection(files, page_ids=ids, max_n_docs=1000)

    train_dataset = dataset.train_split

    valid_dataset = dataset.test_split

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="sequence", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="sequence", config=configuration)

    num_train_steps = len(train_data_loader) * args.epochs
    num_warmup_steps = 0

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
        train_dataloader=train_data_loader, 
        valid_dataloader=None, 
        epochs=args.epochs, 
        configuration=learner, 
        direction=args.direction, 
        measure=args.measure,
        eval_in_train=False
    )

    trainer.execute(write_results=True)

    