from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
from src.dataset.dataset import ParallelDataset
from sentence_transformers import evaluation
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.losses import MSELoss
from sentence_transformers.evaluation import MSEEvaluator
from sentence_transformers import SentenceTransformer
import argparse
from torch.utils.data import DataLoader
import numpy as np
from src.utils.utils import save_file, load_file

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--ep', type=int, dest="epochs", default=1)
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--lr', type=float, dest="lr", default=2e-5)
        parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
        parser.add_argument('--bs', type=int, dest="batch_size", default=16)
        parser.add_argument('--train_path', dest="train_path", type=str, default="../data/jesc/train")
        parser.add_argument('--valid_path', dest="valid_path", type=str, default="../data/jesc/dev")
        parser.add_argument('--save_path', dest="save_path", type=str, default="../models/trained_models/entailment")
        parser.add_argument('--freeze', dest="freeze_weights", type=bool, default=False)
        parser.add_argument('--pretrained', dest="use_pretrained_embeddings", type=bool, default=False)
        parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
        parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=768)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--student', type=str, dest="student_model", default="bert-base-multilingual-cased")
        parser.add_argument('--teacher', type=str, dest="teacher_model", default="bert-base-nli-stsb-mean-tokens")
        parser.add_argument('--setype', type=str, dest="sense_embeddings_type", default="ares_multi")
        parser.add_argument('--pooling', type=str, dest="pooling_strategy", default="avg")
        parser.add_argument('--loss', type=str, dest="loss", default="softmax")
        parser.add_argument('--sense_features', type=bool, dest="senses_as_features", default=True)
        parser.add_argument('--measure', type=str, dest="measure", default="loss")
        parser.add_argument('--direction', type=str, dest="direction", default="minimize")
        parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="trained_models/sbert-jp-jsnli/sbert-jp-jsnli.bin")
        

        args = parser.parse_args()

        student_model = SentenceTransformer(args.student_model)

        teacher_model = SentenceTransformer(args.teacher_model)

        train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model)
        train_data.load_data(args.train_path)
        #train_data_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
        train_data_loader = load_file("../dataset/cached/st-jesc-train-16")
        #save_file(train_data_loader, "../dataset/cached", f"st-jesc-train-{args.batch_size}")

        train_loss = MSELoss(model=student_model)

        dev_dataset = ParallelDataset.build_dataset([args.valid_path]) 

        src_sentences = dev_dataset.get_src_sentences
        tgt_sentences = dev_dataset.get_tgt_sentences

        #NEED TO PASS SRC SENTENCES AND TARGET SENTENCES REFER TO make_multilingual.py
        mse_evaluator = MSEEvaluator(
            source_sentences=src_sentences, 
            target_sentences=tgt_sentences, 
            name="jesc-dev-evaluator", 
            teacher_model=teacher_model, 
            batch_size=args.batch_size
        )

        translation_evaluator = evaluation.TranslationEvaluator(
            src_sentences,
            tgt_sentences,
            name="jesc-translation-evaluator",
            batch_size=args.batch_size,
        )

        student_model.fit(
            train_objectives=[(train_data_loader, train_loss)],
            evaluator = SequentialEvaluator(
                [mse_evaluator, translation_evaluator], 
                main_score_function=lambda scores: np.mean(scores)),
            epochs = args.epochs,
            warmup_steps = 10000,
            evaluation_steps = 1000,
            output_path = args.save_path,
            save_best_model = True,
            use_amp= True,
            optimizer_params = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}
        )
