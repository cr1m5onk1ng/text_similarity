from sentence_transformers.SentenceTransformer import SentenceTransformer
from src.utils.metrics import EmbeddingSimilarityMeter
from src.dataset.sts_dataset import StsDataset
from src.modules.pooling import AvgPoolingStrategy, EmbeddingsPooler, EmbeddingsSimilarityCombineStrategy, SentenceEncodingCombineStrategy
from src.models.losses import SimpleDistillationLoss
from src.evaluation.evaluators import ParaphraseEvaluator
from src.dataset.dataset import SmartParaphraseDataloader
from torch.utils import data
from src.models.sentence_encoder import SiameseSentenceEmbedder
from src.modules.model_compression import SentenceEncoderDistiller
import argparse
from src.dataset.parallel_dataset import *
from src.configurations import config
from src.utils.utils import save_file, load_file
import torch
import transformers

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--ep', type=int, dest="epochs", default=1)
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--bs', type=int, dest="batch_size", default=16)
        parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
        parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=768)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--student_model', type=str, dest="student_model", default="distilbert-base-multilingual-cased")
        parser.add_argument('--teacher_model', type=str, dest="teacher_model", default="sentence-transformers/stsb-distilbert-base")
        parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="trained_models/sencoder-bert-nli-sts")
        parser.add_argument('--max_sentences', type=float, dest="max_sentences", default=300000)
        parser.add_argument('--layers', type=tuple, dest="layers", default=None)
        parser.add_argument('--save_path', dest="save_path", type=str, default="./trained_models")
        parser.add_argument('--save_every_n', dest="save_every_n", type=int, default=50000)

        args = parser.parse_args()
        
        dev_langs_from = ['fr', 'it', 'nl', 'es', 'en']
        dev_langs_to =  ['ar', 'de', 'tr']
        valid_paths = [f"../data/sts/STS2017-extended/STS.{l}-en.txt" for l in dev_langs_from]
        valid_paths += [f"../data/sts/STS2017-extended/STS.en-{l}.txt" for l in dev_langs_to]
        #train_langs = ['ja', 'fr', 'de', 'nl', 'es', 'it']
        train_langs = ['ja']
        train_paths = [f"../data/parallel-sentences/TED2020-en-{l}-train.tsv.gz" for l in train_langs]
        train_dataset = ParallelDataset.build_dataset(train_paths, max_examples=None, skip_header=False)
        #train_dataset.add_dataset("../data/jesc/train", max_examples=args.max_sentences, skip_header=True)
        print(f"Number of training examples: {len(train_dataset)}")
        valid_dataset = StsDataset.build_multilingual(valid_paths)
        
        #teacher_model = SentenceTransformer(args.teacher_model)
        print(f"Loading pretrained model from: {args.pretrained_model_path}")
        teacher_model = SiameseSentenceEmbedder.from_pretrained(args.pretrained_model_path)

        model_config = config.ModelParameters(
            model_name = args.config_name,
            hidden_size = args.embed_dim,
            freeze_weights = False,
            context_layers = (-1,)
        )

        configuration_student = config.ParallelConfiguration(
            model_parameters=model_config,
            model = args.student_model,
            teacher_model=teacher_model,
            sequence_max_len=args.seq_len,
            save_path = args.save_path,
            batch_size = args.batch_size,
            epochs = args.epochs,
            device = torch.device(args.device),
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True),
            tokenizer_student = transformers.AutoTokenizer.from_pretrained(args.student_model, use_fast=True)    
        )

        print("Loading validation dataloader...")
        valid_dataloader = SmartParaphraseDataloader.build_batches(valid_dataset, args.batch_size, mode="standard", config=configuration_student)
        print("Done.")

        print("Loading training dataloader...")
        #train_dataloader = load_file("../dataset/cached/ted_train_multi-dmbert-to-dmbert-bs12")
        #print(f"Data loader sample {train_dataloader[512].src_sentences} ")
        train_dataloader = SmartParaphraseDataloader.build_batches(train_dataset, args.batch_size, mode="parallel", config=configuration_student)
        save_file(train_dataloader, "../dataset/cached", "ted_ja_train-bert-base-nli-sts-to-dmbert")
        print("Done.")

        student_embedder_config = transformers.AutoConfig.from_pretrained(args.student_model)
        student_embedder = transformers.AutoModel.from_pretrained(args.student_model, config=student_embedder_config)

        student_model = SiameseSentenceEmbedder(
            params = configuration_student,
            context_embedder=student_embedder,
            pooling_strategy = AvgPoolingStrategy,
            merge_strategy=SentenceEncodingCombineStrategy,
            loss=SimpleDistillationLoss,
            pooler = EmbeddingsPooler,
        )

        steps = len(train_dataloader) * configuration_student.epochs

        metrics = {"validation": [EmbeddingSimilarityMeter]}
        
        evaluator = ParaphraseEvaluator(
            params = configuration_student,
            model = student_model,
            data_loader = valid_dataloader,
            device = torch.device(args.device),
            metrics = metrics,
            fp16 = args.mixed_precision,
            verbose = True
        )
        
        distiller = SentenceEncoderDistiller(
            config_name=args.config_name,
            params=configuration_student,
            model=student_model,
            teacher=teacher_model,
            layers=args.layers,
            multilingual=True,
            train_dataloader=train_dataloader,
            steps=steps,
            warm_up_steps=10000,
            metrics=None,
            evaluator=evaluator,
            fp16=args.mixed_precision
        )

        distiller.distill(save_every_n=args.save_every_n)