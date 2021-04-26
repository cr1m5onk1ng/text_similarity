import argparse
from src.models.modeling import OnnxTransformerWrapper, TransformerWrapper
from src.models.bert_of_theseus import BertForSequenceClassification, BertModel, BertConfig
from src.models.distilbert_of_theseus import DistilBertForSequenceClassification, DistilBertConfig, DistilBertModel
from src.modules.modules import BertPoolingStrategy, SoftmaxLoss
from src.modules.replacement_scheduler import LinearReplacementScheduler
from src.dataset.dataset import SmartParaphraseDataloader
from src.evaluation.evaluators import Evaluator
import torch
from torch import nn 
import transformers
from sklearn import metrics
from src.utils import utils as utils
from src.configurations.config import ModelParameters, Configuration
from src.utils.metrics import AccuracyMeter
from src.training.learner import Learner
from src.training.train import Trainer
from src.dataset.dataset import Dataset
from src.utils.utils import count_model_parameters
from collections import Counter
import random
import os
from copy import deepcopy


def distill_theseus(args, dataset, labels, metrics, use_wrapper=False):
    
    is_distilbert = "distilbert" in args.model

    model_config = ModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = len(labels),
    )

    configuration = Configuration(
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

    if is_distilbert:
        config = DistilBertConfig.from_pretrained(args.model)
        config.num_labels = len(labels)
        if use_wrapper:
            model = TransformerWrapper(
                params=configuration,
                context_embedder = DistilBertModel.from_pretrained(args.model, config=config),
                pooler = BertPoolingStrategy(configuration),
                loss = SoftmaxLoss(configuration)
            )
            model.context_embedder.transformer.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.context_embedder.transformer.scc_n_layer
            model.context_embedder.transformer.scc_layer = nn.ModuleList([deepcopy(model.context_embedder.transformer.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.context_embedder.transformer
        else:
            model = DistilBertForSequenceClassification.from_pretrained(args.model, config=config)
            model.distilbert.transformer.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.distilbert.transformer.scc_n_layer
            model.distilbert.transformer.scc_layer = nn.ModuleList([deepcopy(model.distilbert.transformer.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.distilbert.transformer
    else:
        config = BertConfig.from_pretrained(args.model)
        config.num_labels = len(labels)
        if use_wrapper:
            model = TransformerWrapper(
                params=configuration,
                context_embedder = BertModel.from_pretrained(args.model, config = config),
                pooler = BertPoolingStrategy(configuration),
                loss = SoftmaxLoss(configuration)
            )
            model.context_embedder.encoder.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.context_embedder.encoder.scc_n_layer
            model.context_embedder.encoder.scc_layer = nn.ModuleList([deepcopy(model.context_embedder.encoder.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.context_embedder.encoder
        else:
            model = BertForSequenceClassification.from_pretrained(args.model, config=config)
            model.bert.encoder.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.bert.encoder.scc_n_layer
            model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.bert.encoder

    train_split, valid_split = dataset.split_dataset(test_perc=0.1)
    if args.dev:
        train_split = train_split[:10000]
        valid_split = valid_split[:10000]
    train_dataset = Dataset(train_split)
    valid_dataset = Dataset(valid_split)
    
    print(f"train dataset size: {len(train_dataset)}")
    print(f"valid dataset size: {len(valid_dataset)}")

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="sequence", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="sequence", config=configuration)

    num_train_steps = len(train_data_loader) * args.epochs
    num_warmup_steps = int(num_train_steps*0.1)

    replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=bert_encoder,
                                                              base_replacing_rate=args.replacing_rate,
                                                              k=args.scheduler_linear_k)

    params_before = count_model_parameters(model) if not use_wrapper else count_model_parameters(model.context_embedder)

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
        eval_in_train=True,
        save_model=False,
        return_model=True
    )

    model = trainer.execute(write_results=True)
    
    if is_distilbert:
        if use_wrapper:
            model.context_embedder.config.n_layers = args.scc_n_layer
            model.context_embedder.transformer.layer = deepcopy(model.context_embedder.transformer.scc_layer)
            model.context_embedder.transformer.scc_layer = None
        else:
            model.config.n_layers = args.scc_n_layer
            model.distilbert.transformer.layer = deepcopy(model.distilbert.transformer.scc_layer)
            model.distilbert.transformer.scc_layer = None
    else:
        if use_wrapper:
            model.context_embedder.config.num_hidden_layers = args.scc_n_layer
            model.context_embedder.encoder.layer = deepcopy(model.context_embedder.encoder.scc_layer)
            model.context_embedder.encoder.scc_layer = None
        else:
            model.config.num_hidden_layers = args.scc_n_layer
            model.bert.encoder.layer = deepcopy(model.bert.encoder.scc_layer)
            model.bert.encoder.scc_layer = None

    params_after = count_model_parameters(model) if not use_wrapper else count_model_parameters(model.context_embedder)

    print(f"Number of parameters before and after layers distillation: Before: {params_before} After: {params_after}")
    print(f"Model config {model.config}")

    if use_wrapper:
        compressed_model = model.context_embedder
        output = deepcopy(model.loss.classifier)
        pooler = deepcopy(model.pooler)
    else:
        compressed_model = model
        output = deepcopy(model.classifier)
        if is_distilbert:
            pooler = deepcopy(model.pre_classifier)
        else:
            pooler = deepcopy(model.bert.pooler.dense)

    if is_distilbert:
        new_context_embedder = transformers.DistilBertModel.from_pretrained(args.model, config=compressed_model.config, state_dict = compressed_model.state_dict())
    else:
        new_context_embedder = transformers.BertModel.from_pretrained(args.model, config=compressed_model.config, state_dict = compressed_model.state_dict())

    new_transformer = OnnxTransformerWrapper(
        params = configuration,
        context_embedder = new_context_embedder,
        pooler = pooler,
        output = output
    )

    saved_params = count_model_parameters(new_context_embedder)
    print(f"Saved model number of paramaters: {saved_params}")
    save_path = os.path.join(args.save_path, f"{args.config_name}-theseus-{args.scc_n_layer}layers")
    print(f"Saving model in {save_path}")
    new_transformer.save_pretrained(save_path)
    return new_transformer, configuration, valid_data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=1)
    parser.add_argument('--name', type=str, dest="config_name", required=True)
    parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--save_path', dest="save_path", type=str, default="trained_models/")
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
    parser.add_argument("--dev", type=bool, dest="dev", default=False)
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

    model, configuration, valid_dataloader = distill_theseus(args, dataset, LABELS_TO_ID, metrics, use_wrapper=True)

    
    evaluator = Evaluator(
        config_name="evaluate classifier",
        params = configuration,
        model = model,
        metrics = metrics,
        fp16 = True,
        verbose = True
    )
    
    print("Evaluating compressed model")
    evaluator.evaluate(valid_dataloader)

    
