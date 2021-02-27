
from src.dataset.dataset import SmartParaphraseDataloader
from src.dataset.parallel_dataset import ParallelDataset
import argparse
from src.configurations import config
import torch
import transformers
from src.utils.utils import save_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=int, dest="epochs", default=1)
    parser.add_argument('--name', type=str, dest="config_name")
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=768)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--student_model', type=str, dest="student_model", default="distilbert-base-multilingual-cased")
    parser.add_argument('--teacher_model', type=str, dest="teacher_model", default="distilroberta-base")
    parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="trained_models/sencoder-bert-nli-sts")
    parser.add_argument('--max_sentences', type=float, dest="max_sentences", default=1200000)
    parser.add_argument('--layers', type=tuple, dest="layers", default=None)
    parser.add_argument('--save_path', dest="save_path", type=str, default="./trained_models")
    parser.add_argument('--max', dest="max_examples", type=int, default=1200000)

    args = parser.parse_args()

    train_langs = ['ja']
    train_paths = [f"../data/parallel-sentences/TED2020-en-{l}-train.tsv.gz" for l in train_langs]
    print(f"Building dataset for languages: {train_langs}")
    train_dataset = ParallelDataset.build_dataset(train_paths, max_examples=None)
    print("Done.")
    print(f"Number of examples: {len(train_dataset)}")
    print()

    model_config = config.ModelParameters(
        model_name = args.config_name,
        hidden_size = args.embed_dim,
        freeze_weights = False,
        context_layers = (-1,)
    )

    configuration_student = config.ParallelConfiguration(
        model_parameters=model_config,
        model = args.student_model,
        sequence_max_len=args.seq_len,
        save_path = args.save_path,
        batch_size = args.batch_size,
        epochs = args.epochs,
        device = torch.device(args.device),
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model),
        tokenizer_student = transformers.AutoTokenizer.from_pretrained(args.student_model)    
    )

    print("Creating dataloader. This may take a while")
    train_dataloader = SmartParaphraseDataloader.build_batches(train_dataset, args.batch_size, mode="parallel", config=configuration_student, sbert_format=True)
    save_file(train_dataloader, "./cached", "ted_train_ja_droberta-distillbert-sbertformat")
    print("Done.")

