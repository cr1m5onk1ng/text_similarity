from typing import Optional
from src.modules.model_compression import DistillationStrategy, PruningStrategy, TheseusCompressionDistillation, convert_to_onnx, convert_to_torchscript
from src.models.distilbert_of_theseus import DistilBertForSequenceClassification, DistilBertConfig
from src.dataset.dataset import SmartParaphraseDataloader
from src.dataset.dataset import Dataset
import os
import random
import argparse
import torch
import transformers
from src.utils import utils
from src.configurations.config import Configuration, ModelParameters
from collections import Counter
from src.utils.metrics import AccuracyMeter

class CompressionPipeline():
    """
    Pipeline that applies a series of compression algorithms
    ranging from distillation to pruning and quantization.

    :param params: Data class containing model parameters
    :param distillation strategy: The distillation strategy to apply
    :param pruning_strategy: The pruning strategy to apply
    :param optimize_for_mobile: Whether or not to convert the model to an optimized format (torchscript, tflite)
    :param convert_to_onnx: Whether or not to convert the model to Onnx format

    """
    def __init__(
    self, 
    params: Configuration,
    distillation_strategy: Optional[DistillationStrategy] = None, 
    pruning_strategy: Optional[PruningStrategy] = None, 
    optimize_for_mobile: bool = False,
    convert_to_onnx: bool = False):
        self.params = params
        self.distillation_strategy = distillation_strategy
        self.pruning_strategy = pruning_strategy
        self.optimize_for_mobile = optimize_for_mobile
        self.convert_to_onnx = convert_to_onnx
        if self.optimize_for_mobile:
            self.convert_to_onnx = False
        if self.convert_to_onnx:
            self.optimize_for_mobile = False

    def __call__(self, dummy_input: str=None, quantize: bool=False):
        model = None
        if self.distillation_strategy is not None:
            model = self.distillation_strategy()
        if self.pruning_strategy is not None:
            model = self.pruning_strategy()
        if self.optimize_for_mobile:
            if dummy_input is None:
                dummy_input = "福岡市は公立小中学校などの教員採用で、筆記試験と面接を省く新たな採用方式を2022年から導入する"
            model = convert_to_torchscript(
                model, 
                self.params.tokenizer, 
                dummy_input, 
                os.path.join(self.params.save_path, self.params.model_config.model_name))
        if self.convert_to_onnx:
            convert_to_onnx(model, self.params, quantize=quantize)
        model.save_pretrained(os.path.join(self.params.save_path, "compressed_model.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=1)
    parser.add_argument('--name', type=str, dest="config_name", required=True)
    parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--save_path', dest="save_path", type=str, default="output/")
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

    config = DistilBertConfig.from_pretrained(args.model)

    config.num_labels = len(LABELS_TO_ID)

    model = DistilBertForSequenceClassification.from_pretrained(args.model, config=config)
   
    """
    model = SentenceTransformerWrapper.from_pretrained(
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

    distiller = TheseusCompressionDistillation(
        params = configuration,
        config_name=args.config_name, 
        model=model, 
        train_dataloader = train_data_loader,
        valid_dataloader = valid_data_loader,
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        fp16=args.mixed_precision, 
        metrics=metrics,
        succ_n_layers = args.scc_n_layer,
    )

    
    pipeline = CompressionPipeline(
        params = configuration,
        distillation_strategy=distiller
    )

    pipeline()
    