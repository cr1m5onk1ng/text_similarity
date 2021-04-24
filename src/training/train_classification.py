import argparse
from src.utils.wikipedia_extractor import WikipediaExtractor
from src.models.bert_of_theseus import BertForSequenceClassification, BertConfig
from src.models.distilbert_of_theseus import DistilBertForSequenceClassification, DistilBertConfig
from src.modules.replacement_scheduler import ConstantReplacementScheduler, LinearReplacementScheduler
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from src.dataset.dataset import SmartParaphraseDataloader
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.models.modeling import TransformerWrapper
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
from src.modules.pooling import AvgPoolingStrategy, BertPoolingStrategy
from src.utils.metrics import AccuracyMeter
from src.training.learner import Learner
from src.training.train import Trainer
from src.dataset.documents_dataset import DocumentDataset
from src.dataset.dataset import Dataset
from collections import Counter
import random
from copy import deepcopy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=1)
    parser.add_argument('--name', type=str, dest="config_name", required=True)
    parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--save_path', dest="save_path", type=str, default="../trained_models/")
    parser.add_argument('--pretrained', dest="use_pretrained_embeddings", type=bool, default=False)
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--scc_n_layer', type=int, dest="scc_n_layer", default=4)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bandainamco-mirai/distilbert-base-japanese")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")
    parser.add_argument("--scheduler_linear_k", default=0.0006, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--replacing_rate", type=float, default=0.3,
                        help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    args = parser.parse_args()

    random.seed(43)

    categories = ["business", "culture", "economy", "politics", "society", "sports", "technology", "opinion", "local", "international"]
    paths = [f"../data/articles/nikkei/nikkei_{cat}.json" for cat in categories]
    # Since we are dealing mostly with small/medium sized sentences, we limit the number of tokens to 64
    dataset = utils.load_file("../dataset/cached/nikkei_dataset")#DocumentDataset.from_json(paths, max_n_tokens=64)
    utils.save_file(dataset, "../dataset/cached/", "nikkei_dataset")
    print(f"Total number of documents: {len(dataset)}")
    print()
    label_to_cat = dataset.label_to_id
    print(f"Mapping: {label_to_cat}")
    print()
    string_labels = list(map(lambda x: label_to_cat[x], dataset.labels))
    print(f"Labels distribution: {Counter(string_labels)}")
    print()

    print(f"Dataset size: {len(dataset)}")
    print()
    LABELS_TO_ID = dataset.label_to_id
    print(f"Labels mapping: {LABELS_TO_ID}")
    print()

    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}

    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = len(LABELS_TO_ID),
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

    config = DistilBertConfig.from_pretrained(args.model)

    config.num_labels = len(LABELS_TO_ID)

    model = DistilBertForSequenceClassification.from_pretrained(args.model, config=config)
    if "distilbert" in args.model:
        model.distilbert.transformer.scc_n_layer = args.scc_n_layer
        scc_n_layer = model.distilbert.transformer.scc_n_layer
        model.distilbert.transformer.scc_layer = nn.ModuleList([deepcopy(model.distilbert.transformer.layer[ix]) for ix in range(scc_n_layer)])
    else:
        model.bert.encoder.scc_n_layer = args.scc_n_layer
        scc_n_layer = model.bert.encoder.scc_n_layer
        model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
    """
    model = SentenceTransformerWrapper.load_pretrained(
        path = args.model,
        params = configuration,
        merge_strategy = SentenceEncodingCombineStrategy(),
        loss = SoftmaxLoss(configuration),
        parallel_mode=False
    )
    

    model = TransformerWrapper(
        context_embedder = AutoModel.from_pretrained(args.model),
        params = configuration,
        pooler = BertPoolingStrategy(configuration),
        loss = SoftmaxLoss(configuration)
    )
    """

    train_split, valid_split = dataset.split_dataset(test_perc=0.1)
    train_dataset = Dataset(train_split)
    valid_dataset = Dataset(valid_split)
    print(f"train dataset size: {len(train_dataset)}")
    print(f"valid dataset size: {len(valid_dataset)}")

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="sequence", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="sequence", config=configuration)

    num_train_steps = len(train_data_loader) * args.epochs
    num_warmup_steps = int(num_train_steps*0.1)

    if "distilbert" in args.model:
        bert_encoder = model.distilbert.transformer
    else:
        bert_encoder = model.bert.encoder
    replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=bert_encoder,
                                                              base_replacing_rate=args.replacing_rate,
                                                              k=args.scheduler_linear_k)

    learner = Learner(
        params = configuration,
        config_name=args.config_name, 
        model=model, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        fp16=args.mixed_precision, 
        metrics=metrics,
        replacing_rate_scheduler = replacing_rate_scheduler
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
