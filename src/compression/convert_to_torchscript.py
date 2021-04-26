from typing import Any
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from src.modules.model_compression import convert_to_torchscript
from src.models.modeling import OnnxTransformerWrapper
from src.configurations import config as config
import transformers
import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, dest="config_name", required=True)
    parser.add_argument('--save_path', dest="save_path", type=str, default="output/pytorch_mobile")
    parser.add_argument('--pretrained_path', dest="pretrained_path", type=str, default="../training/trained_models/distilbert-japanese-nikkei-theseus-theseus-4layers")

    args = parser.parse_args()

    tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_path)

    model_config = config.ModelParameters(
        model_name = "args.config_name",
        hidden_size = 768,
        num_classes = 10,
        use_pretrained_embeddings = False,
        freeze_weights = False,
        context_layers = (-1,)
    )

    configuration = config.Configuration(
        model_parameters=model_config,
        model = "",
        save_path = "args.save_path",
        sequence_max_len = 128,
        dropout_prob = 0.1,
        lr = 2e-5,
        batch_size = 16,
        epochs = 1,
        device = torch.device("cpu"),
        tokenizer = tokenizer,
    )

    
    model = transformers.DistilBertForSequenceClassification.from_pretrained(args.pretrained_path)
    model.eval()

    input = "福岡市は公立小中学校などの教員採用で、筆記試験と面接を省く新たな採用方式を2022年から導入する"

    convert_to_torchscript(model, tokenizer, input, os.path.join(args.save_path, args.config_name))