from src.dataset.distillation_dataset import DistillationDataset
from src.dataset.entailment_dataset import EntailmentDataset
from src.utils.metrics import EmbeddingSimilarityMeter
from src.dataset.sts_dataset import StsDataset
from src.modules.pooling import SentenceEncodingCombineStrategy
from src.models.losses import SimpleDistillationLoss
from src.evaluation.evaluators import ParaphraseEvaluator
from src.dataset.dataset import SmartParaphraseDataloader
from torch.utils import data
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.modules.model_compression import FastFormersDistiller, SentenceEncoderDistiller
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
        parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=128)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--student_model', type=str, dest="student_model", default="sentence-transformers/quora-distilbert-multilingual")
        parser.add_argument('--teacher_model', type=str, dest="teacher_model", default="sentence-transformers/quora-distilbert-multilingual")
        parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="trained_models/sencoder-bert-nli-sts")
        parser.add_argument('--fastformers', type=bool, dest="fastformers_distillation", default=False)
        parser.add_argument('--max_sentences', type=int, dest="max_sentences", default=300000)
        parser.add_argument('--layers', type=list, dest="layers", default=[1, 3, 4, 5])
        parser.add_argument('--save_path', dest="save_path", type=str, default="./output")
        parser.add_argument('--save_every_n', dest="save_every_n", type=int, default=50000)
        parser.add_argument('--state_loss_ratio', dest="state_loss_ratio", type=float, default=0.1)
        parser.add_argument('--att_loss_ratio', dest="att_loss_ratio", type=float, default=0.1)

        args = parser.parse_args()

        dev_langs_from = ['fr', 'it', 'nl', 'es', 'en']
        dev_langs_to =  ['ar', 'de', 'tr']
        valid_paths = [f"../data/sts/STS2017-extended/STS.{l}-en.txt" for l in dev_langs_from]
        valid_paths += [f"../data/sts/STS2017-extended/STS.en-{l}.txt" for l in dev_langs_to]
        train_dataset = EntailmentDataset.build_dataset("../data/nli/AllNLI.tsv", max_examples=args.max_sentences)
        train_dataset.add_dataset("../data/jsnli/train_w_filtering.tsv", max_examples=args.max_sentences)
        distillation_dataset = DistillationDataset.build_dataset(train_dataset, only_src=True)
        print(f"Number of training examples: {len(train_dataset)}")
        valid_dataset = StsDataset.build_multilingual(valid_paths)
       
        model_config = config.ModelParameters(
        model_name = args.config_name,
        hidden_size = args.embed_dim,
        freeze_weights = False,
        context_layers = (-1,)
        )

        configuration = config.ParallelConfiguration(
            model_parameters=model_config,
            model = args.student_model,
            sequence_max_len=args.seq_len,
            save_path = args.save_path,
            batch_size = args.batch_size,
            epochs = args.epochs,
            device = torch.device(args.device),
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True),
            tokenizer_student = transformers.AutoTokenizer.from_pretrained(args.student_model, use_fast=True)    
        )


        print("Loading validation dataloader...")
        valid_dataloader = SmartParaphraseDataloader.build_batches(valid_dataset, args.batch_size, mode="standard", config=configuration)
        print("Done.")

        print("Loading training dataloader...")
        #train_dataloader = load_file("../dataset/cached/distillation-nli-jnli-dmbert")
        train_dataloader = SmartParaphraseDataloader.build_batches(distillation_dataset, args.batch_size, mode="distillation", config=configuration)
        #save_file(train_dataloader, "../dataset/cached", "distillation-nli-jsnli-dmbert")
        print("Done.")
        
        merge_strategy = SentenceEncodingCombineStrategy
        loss = SimpleDistillationLoss

        teacher_model=SentenceTransformerWrapper.load_pretrained(
            args.teacher_model,
            params=configuration,
        )

        student_model = SentenceTransformerWrapper.load_pretrained(
            args.student_model,
            params=configuration,
            loss=SimpleDistillationLoss(params=configuration, teacher_model=teacher_model)
        )

        steps = len(train_dataloader) * configuration.epochs

        metrics = {"validation": [EmbeddingSimilarityMeter]}
        
        evaluator = ParaphraseEvaluator(
            params = configuration,
            model = student_model,
            data_loader = valid_dataloader,
            device = torch.device(args.device),
            metrics = metrics,
            fp16 = args.mixed_precision,
            verbose = True
        )

        print(f"Starting distillation of model: {args.student_model} from teacher model: {args.teacher_model}")
        if args.fastformers_distillation:
            distiller = FastFormersDistiller(
                config_name=args.config_name,
                params=configuration,
                model=student_model,
                teacher=teacher_model,
                state_loss_ratio = args.state_loss_ratio,
                att_loss_ratio = args.att_loss_ratio,
                use_cosine_sim=False,
                train_dataloader=train_dataloader,
                steps=steps,
                warm_up_steps=0,
                metrics=None,
                evaluator=evaluator,
                fp16=args.mixed_precision
            )
        else:
            distiller = SentenceEncoderDistiller(
                config_name=args.config_name,
                params=configuration,
                model=student_model,
                teacher=teacher_model,
                layers=args.layers,
                train_dataloader=train_dataloader,
                steps=steps,
                warm_up_steps=0,
                metrics=None,
                evaluator=evaluator,
                fp16=args.mixed_precision
            )
        reduce_sentences = distillation_dataset.sentences[:20000]
        random.shuffle(reduce_sentences)
        distiller.distill(save_every_n=args.save_every_n, reduce_dim=True, reduce_sentences=reduce_sentences)