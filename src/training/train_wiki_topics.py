import argparse
from src.utils.wikipedia_extractor import WikipediaExtractor
from transformers import BertForSequenceClassification
from transformers import AutoConfig, AutoModel
from src.dataset.dataset import Dataset, SmartParaphraseDataloader
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
from src.dataset.wikipedia_dataset import WikipediaDataset, CATEGORIES, CATEGORIES_EN
from src.dataset.dataset import CrossValidationDataset
from collections import Counter
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=4)
    parser.add_argument('--name', type=str, dest="config_name", required=True)
    parser.add_argument('--lr', type=float, dest="lr", default=1e-4)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--save_path', dest="save_path", type=str, default="../trained_models/")
    parser.add_argument('--pretrained', dest="use_pretrained_embeddings", type=bool, default=False)
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bert-base-multilingual-cased")
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")
    #parser.add_argument('--n_splits', type=int, dest="n_splits", default=3, required=False)
    #parser.add_argument('--fold', type=int, dest="fold", default=0, required=True)

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

    #print("Searching page ids...")
    #ids = extractor.extract_ids_from_categories(CATEGORIES, max_pages=500)
    #print("Done.")
    #print()
    #print("Building dataset.")
    #files = list(utils.search_files("../data/wikipedia-dump/japanese/extracted/"))
    #dataset = WikipediaDataset.from_collection(files, page_ids=ids, max_n_docs=None, max_n_tokens=64)
    #utils.save_file(dataset, "../dataset/cached/", "wikipedia-topics")
    dataset = utils.load_file("../dataset/cached/wikipedia-topics")
    #print("Done")
    
    print(f"Dataset size {len(dataset)}")
    """
    folds = CrossValidationDataset.create_folds(dataset, n_splits=args.n_splits)
    train_dataset = folds.train_splits[args.fold]
    valid_dataset = folds.test_splits[args.fold]
    """
    train_split, valid_split = dataset.split_dataset(test_perc=0.2)
    train_dataset = Dataset(train_split)
    valid_dataset = Dataset(valid_split)
    print(f"train fold size: {len(train_dataset)}")
    print(f"valid fold size: {len(valid_dataset)}")

    print(f"valid dataset examples samples: {valid_dataset.examples[:10]}")
    print(f"valid dataset labels samples: {valid_dataset.labels[:10]}")
    print()
    print(f"Valid dataset labels distribution: {Counter(valid_dataset.labels)}")

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
        valid_dataloader=valid_data_loader, 
        epochs=args.epochs, 
        configuration=learner, 
        direction=args.direction, 
        measure=args.measure,
        eval_in_train=True
    )

    trainer.execute(write_results=True)

    