from transformers import AutoModel
from transformers import AutoTokenizer
from src.dataset.wic_dataset import *
from transformers import AutoTokenizer
from src.modules.pooling import AvgPoolingStrategy, EmbeddingsPooler, EmbeddingsSimilarityCombineStrategy, SentenceBertCombineStrategy
from src.models.sentence_encoder import SentenceTransformerWrapper, SiameseSentenceEmbedder
from src.models.losses import CosineSimilarityLoss, SoftmaxLoss
from src.models.Transformer import Transformer
from src.models.Pooling import Pooling
from src.dataset.distillation_dataset import DistillationDataset
from src.dataset.entailment_dataset import EntailmentDataset
from src.models.SentenceTransformer import SentenceTransformer
from src.utils.metrics import EmbeddingSimilarityMeter
from src.dataset.sts_dataset import StsDataset
from src.dataset.dataset import SmartParaphraseDataloader
import argparse
from src.dataset.parallel_dataset import *
from src.configurations import config
from src.compression.prune import prune_rewire
import torch
from torch.cuda import amp
from tqdm import tqdm

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--ep', type=int, dest="epochs", default=1)
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--bs', type=int, dest="batch_size", default=16)
        parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
        parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=768)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--model', type=str, dest="model", default="sentence-transformers/quora-distilbert-multilingual")
        parser.add_argument('--pretrained', type=str, dest="pretrained_model_path", default="../compression/output/sencoder-dmbert-quora-distilled-2layers")
        parser.add_argument('--target_num_heads', type=int, dest="target_num_heads", default=12)
        parser.add_argument('--target_ffn_dim', type=int, dest="target_ffn_dim", default=600)
        parser.add_argument('--output_dir', dest="output_dir", type=str, default="./output")

        args = parser.parse_args()

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

        model_config = config.ModelParameters(
        model_name = args.config_name,
        hidden_size = args.embed_dim,
        num_classes=3,
        freeze_weights = False,
        context_layers = (-1,)
        )

        configuration = config.ParallelConfiguration(
            model_parameters=model_config,
            model = args.model,
            sequence_max_len=args.seq_len,
            save_path = args.output_dir,
            batch_size = args.batch_size,
            epochs = args.epochs,
            device = torch.device(args.device),
            tokenizer = tokenizer,
        )

        dataset = utils.load_file("../dataset/cached/wikipedia-topics")

        dataloader = SmartParaphraseDataloader.build_batches(dataset, 16, mode="sequence", config=configuration)
        
        
        model = AutoModel(args.model)

        prune_rewire(args, sentence_model, dataloader, tokenizer)