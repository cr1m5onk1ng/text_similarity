from sentence_transformers.evaluation.TranslationEvaluator import TranslationEvaluator
import transformers
import torch
from src.configurations import config
from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
from src.dataset.dataset import ParallelDataset
from sentence_transformers import evaluation
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.losses import MSELoss
from sentence_transformers.evaluation import MSEEvaluator
import sentence_transformers
from sentence_transformers import SentenceTransformer
from src.modules.model_compression import SentenceTransformersDistiller
import argparse
from torch.utils.data import DataLoader
import numpy as np
from src.utils.utils import save_file, load_file
import os

def download_corpora(filepaths):
    if not isinstance(filepaths, list):
        filepaths = [filepaths]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(filepath, "does not exists. Try to download from server")
            filename = os.path.basename(filepath)
            url = "https://sbert.net/datasets/" + filename
            sentence_transformers.util.http_get(url, filepath)

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--ep', type=int, dest="epochs", default=1)
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--bs', type=int, dest="batch_size", default=16)
        parser.add_argument('--train_path', dest="train_path", type=str, default="../data/jesc/train")
        parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/jesc/dev")
        parser.add_argument('--save_path', dest="save_path", type=str, default="./results")
        parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
        parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=768)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--student', type=str, dest="student_model", default="paraphrase-xlm-r-multilingual-v1")
        parser.add_argument('--teacher', type=str, dest="teacher_model", default="paraphrase-xlm-r-multilingual-v1")
        parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="trained_models/sbert-jp-jsnli/sbert-jp-jsnli.bin")
        parser.add_argument('--max_sentences', type=float, dest="max_sentences", default=1200000)
        parser.add_argument('--layers', type=tuple, dest="layers", default=(1, 4, 7, 10))

        args = parser.parse_args()

        student_model = SentenceTransformer(args.student_model)

        teacher_model = SentenceTransformer(args.teacher_model)

        student_model._first_module().auto_model.config.hidden_size = args.embed_dim

        assert(student_model.get_sentence_embedding_dimension()) == args.embed_dim

        train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model)
        train_data.load_data(args.train_path, max_sentences=args.max_sentences)
        train_data_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        #train_data_loader = load_file("../dataset/cached/st-jesc-train-16")
        save_file(train_data_loader, "../dataset/cached", f"st-jesc-train-{args.batch_size}")

        dev_dataset = ParallelDataset.build_dataset([args.valid_path]) 

        src_sentences = dev_dataset.get_src_sentences
        tgt_sentences = dev_dataset.get_tgt_sentences

        if student_model.get_sentence_embedding_dimension() < teacher_model.get_sentence_embedding_dimension():
            reduce_sentences = tgt_sentences[:25000]
        else:
            reduce_sentences = None

        mse_evaluator = MSEEvaluator(
            src_sentences,
            tgt_sentences,
            teacher_model = teacher_model,
            batch_size=args.batch_size
        )

        translation_evaluator = TranslationEvaluator(
            src_sentences,
            tgt_sentences,
            name="translation-eval",
            batch_size=args.batch_size
        )

        model_config = config.SenseModelParameters(
            model_name = args.config_name,
            hidden_size = args.embed_dim
        )

        configuration = config.Configuration(
            model_parameters=model_config,
            model = args.student_model,
            save_path = args.save_path,
            batch_size = args.batch_size,
            epochs = args.epochs,
            device = torch.device(args.device),
            tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base"),
        )   
        
        distiller = SentenceTransformersDistiller(
            params = configuration,
            student_model = student_model,
            teacher_model = teacher_model,
            layers = args.layers,
            train_dataloader=train_data_loader,
            model_save_path=args.save_path,
            evaluators = [mse_evaluator, translation_evaluator]
        )

        distiller.distill(reduce_sentences=reduce_sentences)

