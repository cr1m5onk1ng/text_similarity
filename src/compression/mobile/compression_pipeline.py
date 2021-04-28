from typing import Optional
from src.modules.model_compression import DistillationStrategy, PruningStrategy, TheseusCompressionDistillation, convert_to_onnx, convert_to_torchscript
from src.models.distilbert_of_theseus import DistilBertForSequenceClassification, DistilBertConfig
from src.dataset.dataset import SmartParaphraseDataloader
from src.dataset.dataset import Dataset
import os
import random
import argparse
import torch
import transformers
from src.utils import utils
from src.configurations.config import Configuration, ModelParameters
from collections import Counter
from src.utils.metrics import AccuracyMeter

class CompressionPipeline():
    """
    Pipeline that applies a series of compression algorithms
    ranging from distillation to pruning and quantization.

    :param params: Data class containing model parameters
    :param distillation strategy: The distillation strategy to apply
    :param pruning_strategy: The pruning strategy to apply
    :param optimize_for_mobile: Whether or not to convert the model to an optimized format (torchscript, tflite)
    :param convert_to_onnx: Whether or not to convert the model to Onnx format

    """
    def __init__(
    self, 
    params: Configuration,
    distillation_strategy: Optional[DistillationStrategy] = None, 
    pruning_strategy: Optional[PruningStrategy] = None, 
    optimize_for_mobile: bool = False,
    convert_to_onnx: bool = False):
        self.params = params
        self.distillation_strategy = distillation_strategy
        self.pruning_strategy = pruning_strategy
        self.optimize_for_mobile = optimize_for_mobile
        self.convert_to_onnx = convert_to_onnx
        if self.optimize_for_mobile:
            self.convert_to_onnx = False
        if self.convert_to_onnx:
            self.optimize_for_mobile = False

    def __call__(self, dummy_input: str=None, quantize: bool=False):
        model = None
        if self.distillation_strategy is not None:
            model = self.distillation_strategy()
        if self.pruning_strategy is not None:
            model = self.pruning_strategy()
        if self.optimize_for_mobile:
            if dummy_input is None:
                dummy_input = "福岡市は公立小中学校などの教員採用で、筆記試験と面接を省く新たな採用方式を2022年から導入する"
            model = convert_to_torchscript(
                model, 
                self.params.tokenizer, 
                dummy_input, 
                os.path.join(self.params.save_path, self.params.model_config.model_name))
        if self.convert_to_onnx:
            convert_to_onnx(model, self.params, quantize=quantize)
        model.save_pretrained(os.path.join(self.params.save_path, "compressed_model.pt"))


