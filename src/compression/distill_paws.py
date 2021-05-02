import argparse
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.dataset.paws_dataset import PawsProcessor
from src.modules.modules import SoftmaxLoss,  AvgPoolingStrategy, SentenceBertCombineStrategy
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
    parser.add_argument('--train_path', dest="train_path", type=str, default="../data/paws-x/ja/translated_train.tsv")
    parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/paws-x/ja/dev_2k.tsv")
    parser.add_argument('--save_path', dest="save_path", type=str, default="output/distillation")
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--scc_n_layer', type=int, dest="scc_n_layer", default=3)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="sentence-transformers/quora-distilbert-multilingual")
    parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
    parser.add_argument('--measure', type=str, dest="measure", default="loss")
    parser.add_argument('--direction', type=str, dest="direction", default="minimize")
    parser.add_argument("--scheduler_linear_k", default=0.0006, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--replacing_rate", type=float, default=0.3,
                        help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    parser.add_argument("--dev", type=bool, dest="dev", default=False)

    args = parser.parse_args()

    random.seed(43)

    processor = PawsProcessor()

    train_dataset = processor.build_dataset([args.train_path])

    valid_dataset = processor.build_dataset([args.valid_path])

    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}

    model_config = ModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = 2,
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

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="standard", config=configuration)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="standard", config=configuration)


    compressed_model = distill_theseus(
        args, 
        configuration, 
        train_data_loader, 
        valid_data_loader, 
        num_labels=2, 
        metrics=metrics, 
        use_wrapper=True, 
        sentence_level=True,
        save_for_inference=True)

    """
    evaluator = Evaluator(
        config_name="evaluate classifier",
        params = configuration,
        model = compressed_model,
        metrics = metrics,
        fp16 = True,
        verbose = True
    )
    
    print("Evaluating compressed model")
    evaluator.evaluate(valid_data_loader)
    """
    



    