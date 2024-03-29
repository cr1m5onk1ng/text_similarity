import argparse
from src.dataset.dataset import Dataset, SmartParaphraseDataloader
from src.configurations.config import Configuration, ModelParameters
from src.evaluation.evaluators import Evaluator
from sklearn import metrics
from src.utils import utils as utils
from src.utils.metrics import AccuracyMeter
from src.modules.model_compression import distill_theseus
from collections import Counter
import torch
import transformers
import random


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
    parser.add_argument('--scc_n_layer', type=int, dest="scc_n_layer", default=3)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bandainamco-mirai/distilbert-base-japanese")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")
    parser.add_argument("--scheduler_linear_k", dest="scheduler_linear_k", default=0.0006, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--replacing_rate", type=float, dest="replacing_rate", default=0.3,
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

    train_split, valid_split = dataset.split_dataset(test_perc=0.1)
    if args.dev:
        train_split = train_split[:10000]
        valid_split = valid_split[:10000]
    train_dataset = Dataset(train_split)
    valid_dataset = Dataset(valid_split)
    
    print(f"train dataset size: {len(train_dataset)}")
    print(f"valid dataset size: {len(valid_dataset)}")

    model_config = ModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = len(LABELS_TO_ID),
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

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="sequence", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="sequence", config=configuration)

    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}

    model = distill_theseus(
        args, 
        configuration, 
        train_data_loader, 
        valid_data_loader, 
        len(LABELS_TO_ID), 
        metrics, 
        use_wrapper=True, 
        sentence_level=False)

    
    evaluator = Evaluator(
        config_name="evaluate classifier",
        params = configuration,
        model = model,
        metrics = metrics,
        fp16 = True,
        verbose = True
    )
    
    print("Evaluating compressed model")
    evaluator.evaluate(valid_data_loader)

    
