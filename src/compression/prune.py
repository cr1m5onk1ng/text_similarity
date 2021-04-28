import transformers
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from src.dataset.wic_dataset import *
from transformers import AutoTokenizer
from src.models.modeling import BaseEncoderModel, TransformerWrapper
from src.modules.model_compression import prune_huggingface, prune_rewire
from src.modules.modules import *
from src.utils.metrics import AccuracyMeter, AverageMeter, EmbeddingSimilarityMeter
from src.dataset.sts_dataset import StsDataset
from src.dataset.dataset import SmartParaphraseDataloader
import argparse
from src.dataset.parallel_dataset import *
from src.configurations import config
import torch
from torch.cuda import amp
from tqdm import tqdm

def eval(args, model, eval_dataloader):
    nb_eval_steps = 0
    preds = None
    eval_dataloader = tqdm(eval_dataloader, desc="Computing Head Importance...")
    tot_tokens = 0.0
    accuracy = AccuracyMeter()
    model.to(args.device)
    model.eval()
    for batch in eval_dataloader:
        if isinstance(batch, dict):
            for key in batch:
                el = batch[key]
                if isinstance(el, torch.Tensor):
                    batch[key] = el.to(args.device)
        else:
            batch.to(args.device)
        if isinstance(model, BaseEncoderModel):
            if args.mixed_precision:
                with amp.autocast():
                    outputs = model(features=batch)
            else:
                outputs = model(features=batch)
            tmp_eval_loss = outputs.loss
            logits = outputs.predictions
            labels = batch.labels
        else:
            if not isinstance(batch, dict):
                feats = batch.to_dict()
                labels = batch.labels
            else:
                feats = batch
                labels = batch["labels"]
            if args.mixed_precision:
                with amp.autocast():
                    outputs = model(**feats, labels=batch.labels)
                    
            else:
                outputs = model(**feats, labels=batch.labels)
            tmp_eval_loss = outputs[0]
            logits = outputs[1]
        preds = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        accuracy.update(preds, labels, n=args.batch_size)
        eval_dataloader.set_postfix({"accuracy": "{:.2f}".format(accuracy.avg)})
        nb_eval_steps += 1

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--ep', type=int, dest="epochs", default=1)
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--bs', type=int, dest="batch_size", default=16)
        parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
        parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=768)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--model', type=str, dest="model", default="distilbert-base-multilingual-cased")
        parser.add_argument('--pretrained', type=str, dest="pretrained_model_path", default="../training/trained_models/distilbert-multi-seq-class-nikkei")
        parser.add_argument('--target_num_heads', type=int, dest="target_num_heads", default=6)
        parser.add_argument('--target_ffn_dim', type=int, dest="target_ffn_dim", default=1536)
        parser.add_argument('--output_dir', dest="output_dir", type=str, default="./output")
        parser.add_argument('--normalize', type=bool, dest="normalize_layers", default=False)
        parser.add_argument(
            "--masking_threshold",
            default=0.97,
            type=float,
            help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
        )
        parser.add_argument(
            "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
        )
        parser.add_argument(
            "--dont_normalize_importance_by_layer", 
            dest = "dont_normalize_importance_by_layer",
            action="store_true", 
            help="don't normalize importance score by layers", 
            default=False)

        parser.add_argument(
            "--dont_normalize_global_importance", 
            dest = "dont_normalize_global_importance",
            action="store_true", 
            help="don't normalize importance score by layers", 
            default=False)

        parser.add_argument("--use_huggingface", type=bool, dest="use_huggingface", default=False)

        args = parser.parse_args()

        random.seed(43)

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

        dataset = utils.load_file("../dataset/cached/nikkei_dataset")
        train_split, valid_split = dataset.split_dataset(test_perc=0.1)
        #train_dataset = Dataset(train_split)
        valid_dataset = Dataset(valid_split)
        LABELS_TO_ID = dataset.label_to_id

        model_config = config.SenseModelParameters(
            model_name = args.config_name,
            hidden_size = args.embed_dim,
            num_classes = len(LABELS_TO_ID),
            freeze_weights = False,
            context_layers = (-1,)
        )

        configuration = config.Configuration(
            model_parameters=model_config,
            model = args.model,
            save_path = args.output_dir,
            sequence_max_len = args.seq_len,
            batch_size = args.batch_size,
            epochs = args.epochs,
            device = torch.device(args.device),
            tokenizer = tokenizer,
        )

        valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="sequence", config=configuration)
        autoconfig = AutoConfig.from_pretrained(args.pretrained_model_path, output_attentions=True,)
        autoconfig.num_labels = len(LABELS_TO_ID)
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model_path, config=autoconfig)
        """
        model = TransformerWrapper.load_pretrained(
            args.pretrained_model_path, 
            params=configuration,
            pooler = BertPoolingStrategy(configuration),
            loss = SoftmaxLoss(configuration))
            
        
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
        """

        """
        valid_dataset = EntailmentDataset.build_dataset('../data/nli/AllNLI.tsv', mode="test")
        print()
        print(f"########## Number of examples {len(valid_dataset)} ##################")
        print()
        dataloader = SmartParaphraseDataloader.build_batches(valid_dataset, args.batch_size, mode="standard", config=configuration, sentence_pairs=False)

        sentence_model = SentenceTransformerWrapper.load_pretrained(
            path=args.model,
            params=configuration,
            merge_strategy=SentenceBertCombineStrategy(),
            loss = SoftmaxLoss(params=configuration)
        )
        """
        if args.use_huggingface:
            metrics = {"validation": AccuracyMeter}
            prune_huggingface(args, model, valid_data_loader)
        else:
            model = prune_rewire(args, model, valid_data_loader, tokenizer, is_distilbert=True)
            print(f"Evaluating Pruned Model...")
            eval(args, model, valid_data_loader)
