from transformers import AutoTokenizer
from src.dataset.wic_dataset import *
from transformers import AutoTokenizer
from src.models.sentence_encoder import OnnxSentenceTransformerWrapper, SentenceTransformerWrapper
import argparse
from src.dataset.parallel_dataset import *
from src.configurations import config
import torch
from src.modules.model_compression import convert_to_onnx, quantize_model



if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--model', type=str, dest="model", default="sentence-transformers/bert-base-nli-mean-tokens")
        parser.add_argument('--save_path', dest="save_path", type=str, default="./output")

        args = parser.parse_args()

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        model_config = config.ModelParameters(
        model_name = args.config_name
        )

        configuration = config.Configuration(
            model_parameters=model_config,
            model = args.model,
            save_path = args.save_path,
            tokenizer=tokenizer
        )

        
        sentence_model = OnnxSentenceTransformerWrapper.load_pretrained(
            path=args.model,
            params=configuration
        )

        convert_to_onnx(sentence_model, configuration, quantize=True)
