from src.utils.metrics import EmbeddingSimilarityMeter
from torch._C import device
from src.evaluation.evaluators import ParaphraseEvaluator
from src.dataset.sts_dataset import StsDataset
from src.models.sentence_encoder import SiameseSentenceEmbedder
from src.modules.pooling import *
from src.configurations import config
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import transformers
import torch
import argparse
import numpy as np
from src.logger.logger import logger

if __name__ == "__main__":

    logger = logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--ep', type=int, dest="epochs", default=4)
    parser.add_argument('--name', type=str, dest="config_name")
    parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--save_path', dest="save_path", type=str, default="../evaluation/results")
    parser.add_argument('--pretrained_path', dest="pretrained_path", type=str, default="../training/trained_models/sencoder-bert-nli-sts-paws-contrastive")
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="bert-base-cased")

    args = parser.parse_args()

    valid_dataset = StsDataset.build_dataset("../data/sts/stsbenchmark.tsv", mode="test")

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
        batch_size = args.batch_size,
        epochs = args.epochs,
        device = torch.device(args.device),
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model),    
    )

    print("Creating batches. This may take a while.")
    valid_dataloader = SmartParaphraseDataloader.build_batches(valid_dataset, args.batch_size, mode="standard", config=configuration)
    print("Done.")

    embedder_config = transformers.AutoConfig.from_pretrained(configuration.model)
    embedder = transformers.AutoModel.from_pretrained(configuration.model, config=embedder_config)

    model = SiameseSentenceEmbedder(
        params = configuration,
        context_embedder=embedder,
        loss = CosineSimilarityLoss,
        pooling_strategy = AvgPoolingStrategy,
        pooler = EmbeddingsPooler,
        merge_strategy = EmbeddingsSimilarityCombineStrategy
    )

    model.load_pretrained(args.pretrained_path)

    metrics = {"validation": [EmbeddingSimilarityMeter]}

    evaluator = ParaphraseEvaluator(
         params = configuration,
         model = model,
         data_loader = valid_dataloader,
         device = args.device,
         metrics = metrics,
         fp16 = args.mixed_precision,
         verbose = True,
         return_predictions= False
    )
    print(f"Running evaluation for model: {configuration.model_parameters.model_name}")
    evaluator.evaluate()






