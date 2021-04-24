import argparse
from src.evaluation.evaluators import ClassificationEvaluator
from src.dataset.dataset import SmartParaphraseDataloader
from src.models.modeling import TransformerWrapper
import torch
import transformers
from src.utils import utils as utils
from src.configurations import config as config
from src.models.losses import SoftmaxLoss
from src.modules.pooling import BertPoolingStrategy
from src.utils.metrics import AccuracyMeter
from src.dataset.dataset import Dataset
from collections import Counter
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=1)
    parser.add_argument('--name', type=str, dest="config_name", required=True)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--save_path', dest="save_path", type=str, default="../trained_models/")
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="distilbert-base-multilingual-cased")
    parser.add_argument('--pretrained_model', type=str, dest="pretrained_model", default="../compression/output/distilbert-base-multi-seq-enc-pruned_6_1536")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")

    args = parser.parse_args()

    random.seed(43)

    categories = ["business", "culture", "economy", "politics", "society", "sports", "technology", "opinion", "local", "international"]
    paths = [f"../data/articles/nikkei/nikkei_{cat}.json" for cat in categories]
    # Since we are dealing mostly with small/medium sized sentences, we limit the number of tokens to 64
    dataset = utils.load_file("../dataset/cached/nikkei_dataset")#DocumentDataset.from_json(paths, max_n_tokens=64)
    #utils.save_file(dataset, "../dataset/cached/", "nikkei_dataset")
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

    metrics = {"validation": [AccuracyMeter]}

    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = len(LABELS_TO_ID),
        freeze_weights = False,
        context_layers = (-1,)
    )

    configuration = config.Configuration(
        model_parameters=model_config,
        model = args.model,
        save_path = args.save_path,
        sequence_max_len = args.seq_len,
        batch_size = args.batch_size,
        epochs = args.epochs,
        device = torch.device(args.device),
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model),
    )

    model = TransformerWrapper.load_pretrained(
            args.pretrained_model, 
            params=configuration,
            pooler = BertPoolingStrategy(configuration),
            loss = SoftmaxLoss(configuration))
    

    train_split, valid_split = dataset.split_dataset(test_perc=0.1)
    train_dataset = Dataset(train_split)
    valid_dataset = Dataset(valid_split)
    print(f"train dataset size: {len(train_dataset)}")
    print(f"valid dataset size: {len(valid_dataset)}")

    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="sequence", config=configuration)


    evaluator = ClassificationEvaluator(
        params = configuration,
        model = model,
        data_loader = valid_data_loader,
        device = args.device,
        metrics = metrics,
        fp16 = True,
        verbose = True
    )

    evaluator.evaluate()
