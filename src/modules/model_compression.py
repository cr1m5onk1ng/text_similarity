"""
The code in this file is based on the following papers and relative repositories:
Fastformers: https://github.com/microsoft/fastformers
Sentence-Transformers: https://www.sbert.net/
BERT of Theseus: https://github.com/JetRunner/BERT-of-Theseus

This module provides a collection of functions and classes for Transformer-based model compression.

"""

import os
from pathlib import Path
from src.modules.replacement_scheduler import LinearReplacementScheduler
from src.training.train import Trainer
from src.models.modeling import BaseEncoderModel
from src.models.bert_of_theseus import BertConfig, BertForSequenceClassification, BertModel
from src.models.distilbert_of_theseus import DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel
from src.modules.modules import SentenceEncodingCombineStrategy, SoftmaxLoss, BertPoolingStrategy
from src.utils.utils import count_model_parameters
from typing import Dict, Iterable, List, Optional, Any, Union
import numpy as np
from src.configurations.config import Configuration
from src.utils.metrics import AccuracyMeter, AverageMeter, Metrics
from ..models.sentence_encoder import OnnxSentenceTransformerWrapper, SentenceTransformerWrapper
from src.models.modeling import TransformerWrapper, OnnxTransformerWrapper
import torch
from torch import nn
import torch.nn.functional as F
from ..training.learner import Learner
from tqdm import tqdm
from torch.cuda import amp
from heapq import heappush, heappop
from sentence_transformers import models
from sklearn.decomposition import PCA
from onnxruntime.quantization import quantize_dynamic
from torch.utils.mobile_optimizer import optimize_for_mobile
import transformers
from copy import deepcopy

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def distill_theseus(
    args, 
    configuration: Configuration, 
    train_dataloader: Iterable, 
    valid_dataloader: Iterable, 
    num_labels: int, 
    metrics: dict, 
    use_wrapper: bool=False, 
    sentence_level: bool=False, 
    merging_strategy:nn.Module=None) -> nn.Module:
    """
    This function applies a module replacement distillation strategy desribed
    in the paper BERT of Theseus, which stochastically replaces a group of layers
    of the original model (predecessor) with a single layer of a so-called successor model.
    The approach is simple but the results are incredibly good

    :param configuration: The model configuration containing the model's parameters
    :param train_dataloader: The training dataloader containing the training batches
    :param valid_dataloader: The test dataloader containing the validation batches
    :param dataset: The downstream task dataset to fine-tune the model to. For this distillation method, no additional data is needed
    :param num_labels:  The number of labels of the task
    :param metrics: The metics for training and validations
    :param use_wrapper: Whether to use the library wrapper for transformers models or some huggingface model directly
    :param sentence_lebel: Whether the task is a sentence-level task, thus requiring a Bi-Encoder transformer
    :param merging_strategy: The merging-strategy required for sentence-level downstream task, in case a Bi-Encoder model is used

    :returns The compressed model
    """
    
    is_distilbert = "distilbert" in args.model


    if is_distilbert:
        config = DistilBertConfig.from_pretrained(args.model)
        config.num_labels = num_labels
        if use_wrapper:
            if sentence_level:
                model = SentenceTransformerWrapper(
                    params=configuration,
                    context_embedder = DistilBertModel.from_pretrained(args.model, config=config),
                    pooler = BertPoolingStrategy(configuration),
                    merge_strategy = SentenceEncodingCombineStrategy() if merging_strategy is None else merging_strategy,
                    loss = SoftmaxLoss(configuration, parallel_mode=True)
                )
            else:
                model = TransformerWrapper(
                    params=configuration,
                    context_embedder = DistilBertModel.from_pretrained(args.model, config=config),
                    pooler = BertPoolingStrategy(configuration),
                    loss = SoftmaxLoss(configuration)
                )
            model.context_embedder.transformer.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.context_embedder.transformer.scc_n_layer
            model.context_embedder.transformer.scc_layer = nn.ModuleList([deepcopy(model.context_embedder.transformer.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.context_embedder.transformer
        else:
            model = DistilBertForSequenceClassification.from_pretrained(args.model, config=config)
            model.distilbert.transformer.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.distilbert.transformer.scc_n_layer
            model.distilbert.transformer.scc_layer = nn.ModuleList([deepcopy(model.distilbert.transformer.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.distilbert.transformer
    else:
        config = BertConfig.from_pretrained(args.model)
        config.num_labels = num_labels
        if use_wrapper:
            if sentence_level:
                model = SentenceTransformerWrapper(
                    params=configuration,
                    context_embedder = BertModel.from_pretrained(args.model, config=config),
                    pooler = BertPoolingStrategy(configuration),
                    merging_strategy = SentenceEncodingCombineStrategy() if merging_strategy is None else merging_strategy,
                    loss = SoftmaxLoss(configuration, parallel_mode=True)
                )

            else:
                model = TransformerWrapper(
                    params=configuration,
                    context_embedder = BertModel.from_pretrained(args.model, config = config),
                    pooler = BertPoolingStrategy(configuration),
                    loss = SoftmaxLoss(configuration)
                )

            model.context_embedder.encoder.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.context_embedder.encoder.scc_n_layer
            model.context_embedder.encoder.scc_layer = nn.ModuleList([deepcopy(model.context_embedder.encoder.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.context_embedder.encoder
        else:
            model = BertForSequenceClassification.from_pretrained(args.model, config=config)
            model.bert.encoder.scc_n_layer = args.scc_n_layer
            scc_n_layer = model.bert.encoder.scc_n_layer
            model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
            bert_encoder = model.bert.encoder


    num_train_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(num_train_steps*0.1)

    replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=bert_encoder,
                                                              base_replacing_rate=args.replacing_rate,
                                                              k=args.scheduler_linear_k)

    params_before = count_model_parameters(model) if not use_wrapper else count_model_parameters(model.context_embedder)

    learner = Learner(
        params = configuration,
        config_name=args.config_name, 
        model=model, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        fp16=args.mixed_precision, 
        metrics=metrics,
        replacing_rate_scheduler = replacing_rate_scheduler
    )

    trainer = Trainer(
        args.config_name, 
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader, 
        epochs=args.epochs, 
        configuration=learner, 
        direction=args.direction, 
        measure=args.measure,
        eval_in_train=True,
        save_model=False,
        return_model=True
    )

    model = trainer.execute(write_results=True)
    
    if is_distilbert:
        if use_wrapper:
            model.context_embedder.config.n_layers = args.scc_n_layer
            model.context_embedder.transformer.layer = deepcopy(model.context_embedder.transformer.scc_layer)
            model.context_embedder.transformer.scc_layer = None
        else:
            model.config.n_layers = args.scc_n_layer
            model.distilbert.transformer.layer = deepcopy(model.distilbert.transformer.scc_layer)
            model.distilbert.transformer.scc_layer = None
    else:
        if use_wrapper:
            model.context_embedder.config.num_hidden_layers = args.scc_n_layer
            model.context_embedder.encoder.layer = deepcopy(model.context_embedder.encoder.scc_layer)
            model.context_embedder.encoder.scc_layer = None
        else:
            model.config.num_hidden_layers = args.scc_n_layer
            model.bert.encoder.layer = deepcopy(model.bert.encoder.scc_layer)
            model.bert.encoder.scc_layer = None

    params_after = count_model_parameters(model) if not use_wrapper else count_model_parameters(model.context_embedder)

    print(f"Number of parameters before and after layers distillation: Before: {params_before} After: {params_after}")
    print(f"Model config {model.config}")

    if use_wrapper:
        compressed_model = model.context_embedder
        output = deepcopy(model.loss.classifier)
        pooler = deepcopy(model.pooler)
    else:
        compressed_model = model
        output = deepcopy(model.classifier)
        if is_distilbert:
            pooler = deepcopy(model.pre_classifier)
        else:
            pooler = deepcopy(model.bert.pooler.dense)

    if is_distilbert:
        new_context_embedder = transformers.DistilBertModel.from_pretrained(args.model, config=compressed_model.config, state_dict = compressed_model.state_dict())
    else:
        new_context_embedder = transformers.BertModel.from_pretrained(args.model, config=compressed_model.config, state_dict = compressed_model.state_dict())

    if sentence_level:
        new_transformer = SentenceTransformerWrapper(
            params = configuration,
            context_embedder = new_context_embedder,
            pooler = pooler,
            projection = model.projection
        )

    else:
        new_transformer = OnnxTransformerWrapper(
            params = configuration,
            context_embedder = new_context_embedder,
            pooler = pooler,
            output = output
        )

    saved_params = count_model_parameters(new_context_embedder)
    print(f"Saved model number of paramaters: {saved_params}")
    save_path = os.path.join(args.save_path, f"{args.config_name}-theseus-{args.scc_n_layer}layers")
    print(f"Saving model in {save_path}")
    new_transformer.save_pretrained(save_path)
    return new_transformer

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    print("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def convert_to_torchscript(model: torch.nn.Module, tokenizer: Any, dummy_input: str, save_path: str, include_tti: bool=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    inputs = tokenizer(dummy_input, return_tensors='pt')
    model_dynamic_quantized = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    if include_tti:
        traced_model = torch.jit.trace(model_dynamic_quantized, (inputs['input_ids'], inputs['attention_mask'], input['token_type_ids']), strict=False)
    else:
        traced_model = torch.jit.trace(model_dynamic_quantized, (inputs['input_ids'], inputs['attention_mask']), strict=False)
    optimized_traced_model = optimize_for_mobile(traced_model)
    torch.jit.save(optimized_traced_model,  os.path.join(save_path ,"quantized_model.pt"))


def sort_by_importance(weight, bias, importance, num_instances, stride):
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)), )
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)), )
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat(sorted_bias_to_concat) if sorted_bias_to_concat is not None else None

def prune_rewire(args, model: nn.Module, eval_dataloader: Iterable, tokenizer: transformers.AutoTokenizer, is_distilbert: bool=False):
    """
    This function is based on Fastformer's implementation of transformer model pruning
    """
    results = {}
    model.to(args.device)
    if isinstance(model, BaseEncoderModel):
        if is_distilbert:
            num_hidden_layers = model.context_embedder.config.n_layers
            intermediate_size = model.context_embedder.config.hidden_dim
            hidden_size = model.context_embedder.config.dim
            layers = model.context_embedder.distilbert.transformer.layer
            num_attention_heads = model.context_embedder.config.n_heads
        else:
            num_hidden_layers = model.context_embedder.config.num_hidden_layers
            intermediate_size = model.context_embedder.config.intermediate_size
            hidden_size = model.context_embedder.config.hidden_size
            layers = model.context_embedder.bert.encoder.layer
            num_attention_heads = model.context_embedder.config.num_attention_heads
    else:
        if is_distilbert:
            num_hidden_layers = model.config.n_layers
            intermediate_size = model.config.hidden_dim
            hidden_size = model.config.dim
            layers = model.distilbert.transformer.layer
            num_attention_heads = model.config.n_heads
        else:
            num_hidden_layers = model.config.num_hidden_layers
            intermediate_size = model.config.intermediate_size
            hidden_size = model.config.hidden_size
            layers = model.bert.encoder.layer
            num_attention_heads = model.config.num_attention_heads

    inter_weights = torch.zeros(num_hidden_layers, intermediate_size, hidden_size).to(args.device)
    inter_biases = torch.zeros(num_hidden_layers, intermediate_size).to(args.device)
    output_weights = torch.zeros(num_hidden_layers, hidden_size, intermediate_size).to(args.device)
    
    head_importance = torch.zeros(num_hidden_layers, num_attention_heads).to(args.device)
    ffn_importance = torch.zeros(num_hidden_layers, intermediate_size).to(args.device)

    if is_distilbert:
        for layer_num in range(num_hidden_layers):
            inter_weights[layer_num] = layers._modules[str(layer_num)].ffn.lin1.weight.detach().to(args.device)
            inter_biases[layer_num] = layers._modules[str(layer_num)].ffn.lin1.bias.detach().to(args.device)
            output_weights[layer_num] = layers._modules[str(layer_num)].ffn.lin2.weight.detach().to(args.device)
    else:
        for layer_num in range(num_hidden_layers):
            inter_weights[layer_num] = layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(args.device)
            inter_biases[layer_num] = layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(args.device)
            output_weights[layer_num] = layers._modules[str(layer_num)].output.dense.weight.detach().to(args.device)

    head_mask = torch.ones(num_hidden_layers, num_attention_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    eval_dataloader = tqdm(eval_dataloader, desc="Computing Head Importance...")
    tot_tokens = 0.0
    accuracy = AccuracyMeter()
    for batch in eval_dataloader:
        model.eval()
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
                    outputs = model(features=batch, return_output=False, head_mask=head_mask)
            else:
                outputs = model(features=batch, return_output=False, head_mask=head_mask)
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
                    outputs = model(**feats, head_mask=head_mask, output_attentions=True, labels=batch.labels)
                    
            else:
                outputs = model(**feats, head_mask=head_mask, output_attentions=True, labels=batch.labels)
            tmp_eval_loss = outputs[0]
            logits = outputs[1]
            

        eval_loss += tmp_eval_loss.mean().item()

        # TODO accumulate? absolute value sum?
        tmp_eval_loss.backward()

        # collect attention confidence scores
        head_importance += head_mask.grad.abs().detach()
        preds = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        accuracy.update(preds, labels, n=args.batch_size)
        eval_dataloader.set_postfix({"accuracy": "{:.2f}".format(accuracy.avg)})
        nb_eval_steps += 1
        # collect gradients of linear layers
        if is_distilbert:
            for layer_num in range(num_hidden_layers):
                ffn_importance[layer_num] += torch.abs(
                    torch.sum(layers._modules[str(layer_num)].ffn.lin1.weight.grad.detach()*inter_weights[layer_num], 1) 
                    + layers._modules[str(layer_num)].ffn.lin1.bias.grad.detach()*inter_biases[layer_num])
        else:
            for layer_num in range(num_hidden_layers):
                ffn_importance[layer_num] += torch.abs(
                    torch.sum(layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach()*inter_weights[layer_num], 1) 
                    + layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach()*inter_biases[layer_num])
        # TODO See if there is a better strategy
        if isinstance(model, BaseEncoderModel):
            if hasattr(batch, "sentence_1_features"):
                tot_tokens += (batch.sentence_1_features.attention_mask.float().detach().sum().data + batch.sentence_2_features.attention_mask.float().detach().sum().data )
            else:
                tot_tokens += batch.to_dict()["attention_mask"].float().detach().sum().data
        else:
            if isinstance(batch, dict):
                attention_mask = batch["attention_mask"]
            else:
                attention_mask = batch.to_dict()["attention_mask"]
            if hasattr(batch, "sentence_1_features"):
                tot_tokens += (attention_mask.float().detach().sum().data + batch.sentence_2_features.attention_mask.float().detach().sum().data )
            else:
                tot_tokens += attention_mask.float().detach().sum().data
        

    head_importance /= tot_tokens

    # Layerwise importance normalization
    if args.normalize_layers:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # rewire the network
    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    num_heads = num_attention_heads
    head_size = hidden_size / num_heads
    
    for layer_num in range(num_hidden_layers):
        # load query, key, value weights
        if is_distilbert:
            query_weight = layers._modules[str(layer_num)].attention.q_lin.weight
            query_bias = layers._modules[str(layer_num)].attention.q_lin.bias
            key_weight = layers._modules[str(layer_num)].attention.k_lin.weight
            key_bias = layers._modules[str(layer_num)].attention.k_lin.bias
            value_weight = layers._modules[str(layer_num)].attention.v_lin.weight
            value_bias = layers._modules[str(layer_num)].attention.v_lin.bias
        else:
            query_weight = layers._modules[str(layer_num)].attention.self.query.weight
            query_bias = layers._modules[str(layer_num)].attention.self.query.bias
            key_weight = layers._modules[str(layer_num)].attention.self.key.weight
            key_bias = layers._modules[str(layer_num)].attention.self.key.bias
            value_weight = layers._modules[str(layer_num)].attention.self.value.weight
            value_bias = layers._modules[str(layer_num)].attention.self.value.bias
        print(f"Query Weight Size Before Sorting: {query_weight.shape} Query Bias Size Before Sorting: {query_bias.shape}")
        print(f"Key Weight Size Before Sorting: {key_weight.shape} Key Bias Size Before Sorting: {key_bias.shape}")
        print(f"Value Weight Size Before Sorting: {value_weight.shape} Value Bias Size Before Sorting: {value_bias.shape}")
        # sort query, key, value based on the confidence scores
        query_weight, query_bias = sort_by_importance(query_weight,
            query_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        print(f"Query Weight Size: {query_weight.shape} Query Bias Size: {query_bias.shape}")
        if is_distilbert:
            layers._modules[str(layer_num)].attention.q_lin.weight = torch.nn.Parameter(query_weight)
            layers._modules[str(layer_num)].attention.q_lin.bias = torch.nn.Parameter(query_bias)
        else:
            layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
            layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = sort_by_importance(key_weight,
            key_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        print(f"Key Weight Size: {query_weight.shape} Key Bias Size: {query_bias.shape}")
        if is_distilbert:
            layers._modules[str(layer_num)].attention.k_lin.weight = torch.nn.Parameter(key_weight)
            layers._modules[str(layer_num)].attention.k_lin.bias = torch.nn.Parameter(key_bias)
        else:
            layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
            layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = sort_by_importance(value_weight,
            value_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        print(f"Value Weight Size: {query_weight.shape} Value Bias Size: {query_bias.shape}")
        if is_distilbert:
            layers._modules[str(layer_num)].attention.v_lin.weight = torch.nn.Parameter(value_weight)
            layers._modules[str(layer_num)].attention.v_lin.bias = torch.nn.Parameter(value_bias)
        else:
            layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
            layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

        # output matrix
        if is_distilbert:
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].attention.out_lin.weight.transpose(0, 1),
                None,
                head_importance[layer_num],
                args.target_num_heads,
                head_size)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].attention.out_lin.weight = torch.nn.Parameter(weight_sorted)
    
            weight_sorted, bias_sorted = sort_by_importance(
                layers._modules[str(layer_num)].ffn.lin1.weight,
                layers._modules[str(layer_num)].ffn.lin1.bias, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            layers._modules[str(layer_num)].ffn.lin1.weight = torch.nn.Parameter(weight_sorted)
            layers._modules[str(layer_num)].ffn.lin1.bias = torch.nn.Parameter(bias_sorted)
    
            # ffn output matrix input side
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].ffn.lin2.weight.transpose(0, 1),
                None, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].ffn.lin2.weight = torch.nn.Parameter(weight_sorted)
        else:
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].attention.output.dense.weight.transpose(0, 1),
                None,
                head_importance[layer_num],
                args.target_num_heads,
                head_size)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)

            weight_sorted, bias_sorted = sort_by_importance(
                layers._modules[str(layer_num)].intermediate.dense.weight,
                layers._modules[str(layer_num)].intermediate.dense.bias, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            layers._modules[str(layer_num)].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
            layers._modules[str(layer_num)].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)

            # ffn output matrix input side
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].output.dense.weight.transpose(0, 1),
                None, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].output.dense.weight = torch.nn.Parameter(weight_sorted)

    # save pruned model
    from pathlib import Path
    path = os.path.join(args.output_dir, args.config_name + "-pruned" + "_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))
    Path(path).mkdir(exist_ok=True)

    model.config.hidden_act = 'relu'    # use ReLU activation for the pruned models.
    if is_distilbert:
        model.config.n_heads = min([num_heads, args.target_num_heads])
        model.config.hidden_dim = layers._modules['0'].ffn.lin1.weight.size(0)
    else:
        model.config.num_attention_heads = min([num_heads, args.target_num_heads])
        model.config.intermediate_size = layers._modules['0'].intermediate.dense.weight.size(0)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    return model

def compute_heads_importance(
    args, 
    model: nn.Module, 
    eval_dataloader: Iterable, 
    head_mask: torch.Tensor=None, 
    compute_importance:bool=True, 
    actually_pruned:bool=False):
    """
    This function is based on Huggingface's implementation of Multi Head Attention pruning 
    """
    accuracy = AccuracyMeter()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    #attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    model.to(args.device)
    model.eval()
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    labels = None
    loss = None
    eval_dataloader = tqdm(eval_dataloader, desc="Computing Head Importance...")
    tot_tokens = 0.0
    for batch in eval_dataloader:
        model.eval()
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
                    outputs = model(features=batch, return_output=False, head_mask=head_mask)        
            else:
                outputs = model(features=batch, return_output=False, head_mask=head_mask)
            loss = outputs.loss
            logits = outputs.predictions
            labels = outputs.labels
        else:
            if not isinstance(batch, dict):
                feats = batch.to_dict()
                labels = batch.labels.to(args.device)
            else:
                feats = batch
                labels = batch["labels"].to(args.device)
            if args.mixed_precision:
                with amp.autocast():
                    outputs = model(**feats, head_mask=head_mask, labels=labels)
                    
            else:
                outputs = model(**feats, head_mask=head_mask, labels=labels)
            loss, logits, all_attentions = (
                outputs[0],
                outputs[1],
                outputs[-1]
            )
        
        # TODO accumulate? absolute value sum?
        loss.backward()

        # collect attention confidence scores
        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # collect gradients of linear layers
        
        # TODO See if there is a better strategy
        if isinstance(model, BaseEncoderModel):
            if hasattr(batch, "sentence_1_features"):
                tot_tokens += (batch.sentence_1_features.attention_mask.float().detach().sum().data + batch.sentence_2_features.attention_mask.float().detach().sum().data )
            else:
                tot_tokens += batch.to_dict()["attention_mask"].float().detach().sum().data
        else:
            if isinstance(batch, dict):
                attention_mask = batch["attention_mask"]
            else:
                attention_mask = batch.to_dict()["attention_mask"]
            if hasattr(batch, "sentence_1_features"):
                tot_tokens += (attention_mask.float().detach().sum().data + batch.sentence_2_features.attention_mask.float().detach().sum().data )
            else:
                tot_tokens += attention_mask.float().detach().sum().data
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        """
        for m in meters.metrics:
            m.update(logits, labels, n=len(batch))
        """
        accuracy.update(preds, labels, n=args.batch_size)
        #eval_dataloader.set_postfix(**meters.set_postfix())
        eval_dataloader.set_postfix({"accuracy": "{:.2f}".format(accuracy.avg)})
    head_importance /= tot_tokens
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())
    print_2d_tensor(head_importance)
    print("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=args.device
    )
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)

    return head_importance, accuracy.avg

def mask_heads(args, model, eval_dataloader):
    head_importance, accuracy = compute_heads_importance(args, model, eval_dataloader)
    original_score = accuracy
    print(f"Pruning: original score: {original_score}, threshold: {original_score * args.masking_threshold}")

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    while current_score >= original_score * args.masking_threshold:
        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        print(f"Heads to mask: {str(current_heads_to_mask.tolist())}")
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        head_importance, new_accuracy = compute_heads_importance(
            args, model, eval_dataloader, head_mask=new_head_mask
        )
        current_score = new_accuracy
        print(
            f"Masking: current score: {current_score}, remaining heads {new_head_mask.sum()} ({new_head_mask.sum() / new_head_mask.numel() * 100}percents)")

    print("Final head mask")
    print_2d_tensor(head_mask)
    #np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask.detach()

def prune_heads(args, model, eval_dataloader, head_mask):
    _, accuracy = compute_heads_importance(
        args, model, eval_dataloader, compute_importance=False, head_mask=head_mask
    )
    score_masking = accuracy
    original_num_params = sum(p.numel() for p in model.parameters())

    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.prune_heads(heads_to_prune)

    pruned_num_params = sum(p.numel() for p in model.parameters())

    _, accuracy = compute_heads_importance(
        args,
        model,
        eval_dataloader,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
    )
    score_pruning = accuracy
    print(f"Original number of params: {original_num_params}")
    print(f"Pruned number of params: {pruned_num_params}")
    print(f"Score masking: {score_masking} Score Pruning: {score_pruning}")

def prune_huggingface(args, model, eval_dataloader):
    head_mask = mask_heads(args, model, eval_dataloader)
    prune_heads(args, model, eval_dataloader, head_mask)
    model.save_pretrained(args.output_dir)

def quantize_model(path, save_path):
    Path(path)
    Path(save_path)

    quantize_dynamic(
        model_input=path,
        model_output=save_path
    )

def convert_to_onnx(
    model: nn.Module, 
    params: Configuration, 
    opset: int=11, 
    quantize: bool=False, 
    has_token_type_ids = False,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]]=None, 
    dynamic_axes: Optional[Dict[int, str]] = None,
    sample_string: Optional[str]=None) -> str:
    """
    This function is based on Fastformer's implementation of transformer model optimization via Onnx conversion
    """
    
    print(f"################## Staring ONNX graph optimization on model: {params.model} ##################")
    if sample_string is None:
        sample_string = "This is a sample input."
    tokens = params.tokenizer.encode_plus(sample_string)
    model.to(torch.device("cpu"))
    model.eval()
    
    if input_names is None:
        if has_token_type_ids:
            input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        else:
            input_names = ['input_ids', 'attention_mask']
    if output_names is None:
        output_names = ['output_0']
    
    print(f"TOKEN TYPE IDS: {tokens['token_type_ids']}")
    if dynamic_axes is None:
        if has_token_type_ids:
            dynamic_axes = {
                'input_ids': {
                    0: 'batch',
                    1: 'sequence'
                },
                'attention_mask': {
                    0: 'batch',
                    1: 'sequence'
                },
                'segment_ids': {
                    0: 'batch',
                    1: 'sequence'
                },
                'output_0': {
                    0: 'batch',
                    1: 'sequence'
                }
            }
        else:
            dynamic_axes = {
                'input_ids': {
                    0: 'batch',
                    1: 'sequence'
                },
                'attention_mask': {
                    0: 'batch',
                    1: 'sequence'
                }, 
                'output_0': {
                    0: 'batch',
                    1: 'sequence'
                }
            }
    if has_token_type_ids:
        model_args = (torch.tensor(tokens['input_ids']).unsqueeze(0),
                  torch.tensor(tokens['attention_mask']).unsqueeze(0),
                  torch.tensor(tokens['token_type_ids']).unsqueeze(0))
    else:
        model_args = (torch.tensor(tokens['input_ids']).unsqueeze(0),
                      torch.tensor(tokens['attention_mask']).unsqueeze(0))
    save_path = params.save_path + f"/{params.model_parameters.model_name}" 
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    onnx_path = save_path + "/model.onnx"
    
    torch.onnx.export(
        model,
        model_args,
        f=onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=opset,
    )
    print(f"Optimized model correctly exported in: {onnx_path}")
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    print("Checking ONNX model loaded from: {}".format(onnx_path))
    try:
        onnx_options = SessionOptions()
        sess = InferenceSession(onnx_path, onnx_options)
        print("Model loaded successfully")
        if has_token_type_ids:
            session_input = {
                'input_ids': [tokens['input_ids']],
                'attention_mask': [tokens['attention_mask']],
                'token_type_ids': [tokens['token_type_ids']]
                }
        else:
            session_input = {
                'input_ids': [tokens['input_ids']],
                'attention_mask': [tokens['attention_mask']]
                }
        output_onnx = sess.run(None, session_input)
        print(output_onnx)
    except RuntimeException as re:
        print("Error while loading the model: {}".format(re))
    
    if quantize:
        quant_path = save_path + "/quantized_model.onnx"
        print(f"Starting 8 bit quantization on model: {params.model_parameters.model_name}")
        quantize_model(onnx_path, quant_path)
        print(f"Quantized model correctly exported in {quant_path}")
    return quant_path if quantize else onnx_path


"""
The classes below are an attempt to organize the model compression pipeline
"""

class PruningStrategy():
    """
    Base model for pruning strategies
    """
    def __init__(self, params: Configuration, model: nn.Module, dataloader: Iterable):
        self.params = params
        self.model = model
        self.dataloader = dataloader


class FastFormersPruningStrategy(PruningStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        is_distilbert = "distilbert" in self.params.model
        return prune_rewire(self.params, self.model, self.dataloader, self.params.tokenizer, is_distilbert)


class DistillationStrategy(Learner):
    """
    Base class for distillation strategies
    """
    def __init__(self, *args, evaluator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator

    def _distillation_step(self, save_every_n=None):
        raise NotImplementedError()

    def distill(self, save_every_n=None, reduce_dim=False, reduce_sentences=None):
        if reduce_dim:
            assert reduce_sentences is not None
            DistillationStrategy.reduce_dim(self.model, reduce_sentences, self.model.params.model_parameters.hidden_size)
            DistillationStrategy.reduce_dim(self.teacher, reduce_sentences, self.teacher.params.model_parameters.hidden_size)

        min_loss = np.inf
        for epoch in range(self.params.epochs):
            print(f"###### EPOCH {epoch+1} #######")
            results = self._distillation_step(save_every_n=save_every_n)
            loss = results["loss"]
            if loss < min_loss:
                min_loss = loss
                self.model.save_pretrained(os.path.join(self.params.save_path, self.config_name))
            if self.evaluator is not None:
                self.evaluator.evaluate()

    @staticmethod
    def reduce_dim(model, sentences, dim=128):
        embeddings = model.encode_text(sentences, output_np=True)
        pca = PCA(n_components=dim)
        pca.fit(embeddings)
        components = np.asarray(pca.components_)
        if not hasattr(model, "projection"):
            model.projection = nn.Linear(model.embedding_size, dim)
        model.projection.weight = torch.nn.Parameter(torch.tensor(components))

    @staticmethod
    def reduce_teacher_dim(sentences, student, teacher):
        logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
        embeddings = teacher.encode_text(sentences, output_np=True)
        pca = PCA(n_components=student.get_sentence_embedding_dimension())
        pca.fit(embeddings)

        #Add projection layer to teacher that projects the embeddings down to the student embedding size
        assert hasattr(teacher, "projection")
        teacher.projection.weight = torch.nn.Parameter(torch.tensor(pca.components_))


class TheseusCompressionDistillation(DistillationStrategy):
    """
    Distillation strategy based on BERT of Theseus implementation.
    """
    def __init__(
        self, 
        train_dataloader,
        valid_dataloader,
        *args, 
        replacing_rate_scheduler=None,
        succ_n_layers: int=6,
        scheduler_linear_k: float = 0.0006,
        replacing_rate = 0.3,
         **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.succ_n_layers = succ_n_layers
        
        if "distilbert" in self.params.model:
            self.model.distilbert.transformer.scc_n_layer = self.succ_n_layers
            self.model.distilbert.transformer.scc_layer = nn.ModuleList([deepcopy(self.model.distilbert.transformer.layer[ix]) for ix in range(self.succ_n_layers)])
            bert_encoder = self.model.distilbert.transformer
        else:
            self.model.bert.encoder.scc_n_layer = self.succ_n_layers
            self.model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(self.model.bert.encoder.layer[ix]) for ix in range(self.succ_n_layers)])
            bert_encoder = self.model.bert.encoder
        
        self.replacing_rate_scheduler = replacing_rate_scheduler
        if self.replacing_rate_scheduler is None:
            self.replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=bert_encoder,
                                                              base_replacing_rate=replacing_rate,
                                                              k=scheduler_linear_k)

    def __call__(self) -> nn.Module:
        params_before = 0 
        if isinstance(self.model, BaseEncoderModel):
            params_before = count_model_parameters(self.model.context_embedder)
        else:
            params_before = count_model_parameters(self.model)

        trainer = Trainer(
            self.config_name, 
            train_dataloader=self.train_dataloader, 
            valid_dataloader=self.valid_dataloader, 
            epochs=self.params.epochs, 
            configuration=self, 
            direction="maximize", 
            measure="accuracy",
            eval_in_train=self.valid_dataloader is not None,
            save_model=False,
            return_model=True
        )
        model = trainer.execute(write_results=False)
        params_after = 0
        if isinstance(self.model, BaseEncoderModel):
            if "distilbert" in self.params.model:
                model.context_embedder.config.n_layers = self.succ_n_layers
                model.context_embedder.distilbert.transformer.layer = None#deepcopy(self.model.context_embedder.distilbert.transformer.scc_layer)
                #model.context_embedder.distilbert.transformer.scc_layer = None  
            else:
                model.context_embedder.config.num_hidden_layers = self.succ_n_layers
                model.context_embedder.bert.encoder.layer = deepcopy(self.model.context_embedder.bert.encoder.scc_layer)
                model.context_embedder.bert.encoder.scc_layer = None
            params_after = count_model_parameters(model.context_embedder)
        else:
            if "distilbert" in self.params.model:
                model.config.n_layers = self.succ_n_layers
                model.distilbert.transformer.layer = None#deepcopy(self.model.distilbert.transformer.scc_layer)
                #model.distilbert.transformer.scc_layer = None
            else:
                model.config.num_hidden_layers = self.succ_n_layers
                model.bert.encoder.layer = None#deepcopy(self.model.bert.encoder.scc_layer)
                #model.bert.encoder.scc_layer = None
            params_after = count_model_parameters(model)
        print(f"Number of parameters before and after: Before: {params_before} After: {params_after}")
        save_path = os.path.join(self.params.save_path, f"{self.config_name}-theseus-{self.succ_n_layers}layers")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)
        return model


class SentenceEncoderDistiller(DistillationStrategy):
    """
    Distiller module based on SBERT implementation.
    The distillation consists in a simple module dropping.
    Some layers of the models are dropped while others are retained.
    For 12-layers models, dropping 4 layers seems to lead to a very
    slight decrease in performance for sentence-level similarity tasks
    while cutting inference time by a consistent amount
    """
    def __init__(
        self, 
        teacher: nn.Module,
        dataloader,
        layers: list,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.dataloader = dataloader
        if isinstance(self.model, SentenceTransformerWrapper):
            model = self.model.context_embedder
        else:
            model = self.model
        if layers is not None:
            print(f"Number of parameters before layers removal: {self.model.params_num}")
            if "distilbert" in self.params.model:
                layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.transformer.layer) if i in layers])
                model.transformer.layer = layers_to_keep
                model.config.n_layers = len(layers_to_keep)
                self.model.context_embedder = model
            else:
                layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.encoder.layer) if i in layers])
                model.encoder.layer = layers_to_keep
                model.config.num_hidden_layers = len(layers_to_keep)
                self.model.context_embedder = model
            print(f"Number of parameters after layers removal: {self.model.params_num}")
            assert self.model.context_embedder.config.num_hidden_layers == len(layers)

    def _step(self, data, b_idx):
        output_student = self.model(data, parallel_mode=False)
        loss = output_student.loss  
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (b_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
        return loss, output_student.predictions

    def _mixed_precision_step(self, data, b_idx):
        with amp.autocast():
            output_student = self.model(data, parallel_mode=False)
        loss = output_student.loss  
        scale_before_step = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
  
        return loss, output_student.predictions, scale_before_step

    def _distillation_step(self, save_every_n=None):
        logging.info(f"##### Making some fine liquor with model: {self.config_name}#####")
        losses = AverageMeter("loss")
        if self.metrics is not None:
            meters = Metrics(*self.metrics["training"])
        else:
            meters = None
        iterator = tqdm(self.dataloader, total=len(self.train_dataloader))
        self.model.to(self.params.device)
        self.teacher.to(self.params.device)
        self.teacher.eval()
        self.model.train()
        results = []
        skip_scheduler = False
        for b_idx, data in enumerate(iterator):
            data.to(self.params.device)
            if self.fp16:
                loss, embeddings, scale_before_step = self._mixed_precision_step(data, b_idx)
                skip_scheduler = self.scaler.get_scale() != scale_before_step
            else:
               loss, embeddings = self._step(data, b_idx)

            if (b_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                if not skip_scheduler:
                    self.scheduler.step()
        
            losses.update(loss.item(), self.params.batch_size)
            if meters is not None:
                    labels = data.labels.cpu().numpy()
                    if embeddings is not None:
                        embeddings = embeddings.detach().cpu().numpy()
                        for m in meters.metrics:
                            m.update(embeddings, labels, n=self.params.batch_size)
                    iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
            if meters is not None:
                iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
            else:
                iterator.set_postfix({"loss": "{:.2f}".format(losses.avg)})
            if save_every_n is not None:
                if (b_idx+1) % save_every_n == 0:
                    self.save_model(os.path.join(self.params.save_path, self.config_name))
        iterator.close()
        if self.verbose and meters is not None:
            meters.display_metrics()
        results = {"loss": losses.avg}
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                m.reset()
        return results


class FastFormersDistiller(DistillationStrategy):
    def __init__(self, state_loss_ratio, att_loss_ratio, use_cosine_sim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_loss_ratio = state_loss_ratio
        self.att_loss_ratio = att_loss_ratio
        self.use_cosine_sim = use_cosine_sim
        self.loss_mse = nn.MSELoss()
        self.loss_cs = nn.CosineSimilarity(dim=2)
        self.loss_cs_att = nn.CosineSimilarity(dim=3)
        self.tr_att_loss = AverageMeter("tr_att_loss")
        self.tr_rep_loss = AverageMeter("tr_rep_loss")
        self.tr_cls_loss = AverageMeter("tr_cls_loss")
        self.cls_loss = 0.
        self.rep_loss = 0.
        self.attn_loss = 0.
        self.teacher_layer_num = self.teacher.config.num_hidden_layers
        self.model_layer_num = self.model.config.num_hidden_layers

    def _step(self, data, b_idx):
        with torch.no_grad():
            pooled_teacher, outputs_teacher = self.teacher.encode(data, return_output=True)
        pooled_student, outputs_student = self.model.encode(data, return_output=True)
        loss = self._calculate_losses(outputs_teacher, outputs_student)
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (b_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
        return loss, torch.stack([pooled_student, pooled_teacher], dim=0)

    def _mixed_precision_step(self, data, b_idx):
        with torch.no_grad():
            pooled_teacher, outputs_teacher = self.teacher.encode(data, return_output=True)
        pooled_student, outputs_student = self.model.encode(data, return_output=True)
        loss = self._calculate_losses(outputs_teacher, outputs_student)
        scale_before_step = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, torch.stack([pooled_student, pooled_teacher], dim=0), scale_before_step

    def _logits_distillation(self, outputs_teacher, outputs_student):
        kd_loss = self.soft_cross_entropy(outputs_student[1], outputs_teacher[1])
        loss = kd_loss
        self.cls_loss += kd_loss
        self.tr_cls_loss.update(kd_loss.item(), n=self.train_dataloader.get_batch_size)
        return loss

    def _embeddings_distillation(self, outputs_teacher, outputs_student):
        teacher_reps = outputs_teacher[2]
        student_reps = outputs_student[2]
        new_teacher_reps = [teacher_reps[0].detach_(), teacher_reps[self.teacher_layer_num].detach_()]
        new_student_reps = [student_reps[0].detach_(), student_reps[self.model_layer_num].detach_()]
        tmp_loss = 0
        for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
            if self.use_cosine_sim:
                tmp_loss = 1.0 - self.loss_cs(student_rep, teacher_rep).mean()
            else:
                tmp_loss = self.loss_mse(student_rep, teacher_rep)
        self.rep_loss += tmp_loss
        self.tr_rep_loss.update(tmp_loss.item(), n=self.train_dataloader.get_batch_size)
        return self.state_loss_ratio * self.rep_loss

    def _attention_distillation(self, outputs_teacher, outputs_student):
        teacher_atts = outputs_teacher[3]
        student_atts = outputs_student[3]
        assert self.teacher_layer_num == len(teacher_atts)
        assert self.model_layer_num == len(student_atts)
        assert self.teacher_layer_num % self.model_layer_num == 0
        layers_per_block = int(self.teacher_layer_num / self.model_layer_num)
        new_teacher_atts = [teacher_atts[i*layers_per_block + layers_per_block - 1] for i in range(self.model_layer_num)]
        tmp_loss = 0
        for student_att, teacher_att in zip(student_atts, new_teacher_atts):
            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.params.device), student_att)
            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.params.device), teacher_att)
            tmp_loss = 1.0 - self.loss_cs_att(student_att, teacher_att).mean()
        self.attn_loss += tmp_loss
        self.tr_att_loss.update(tmp_loss.item(), n=self.train_dataloader.get_batch_size)
        return self.att_loss_ratio * self.attn_loss

    def _calculate_losses(self, outputs_teacher, outputs_student):
        loss = self._logits_distillation(outputs_teacher, outputs_student)
        loss += self._embeddings_distillation(outputs_teacher, outputs_student)
        #loss += self._attention_distillation(outputs_teacher, outputs_student)
        return loss
    
    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).sum(dim=-1).mean() 

    def _distillation_step(self, save_every_n=None):
        losses = AverageMeter("loss")
        if self.metrics is not None:
            meters = Metrics(*self.metrics["training"])
        else:
            meters = None
        iterator = tqdm(self.train_dataloader, total=len(self.train_dataloader))
        self.model.to(self.params.device)
        self.teacher.to(self.params.device)
        self.model.train()
        self.teacher.eval()
        loss = 0.
        for b_idx, data in enumerate(iterator):
            data.to(self.params.device) 
            skip_scheduler = False
            if self.fp16:
                loss, embeddings, scale_before_step = self._mixed_precision_step(data, b_idx)
                skip_scheduler = self.scaler.get_scale() != scale_before_step
            else:
               loss, embeddings = self._step(data, b_idx)
            
            if self.scheduler is not None:
                if not skip_scheduler:
                    self.scheduler.step()
            if (b_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.zero_grad()
            losses.update(loss.item(), self.params.batch_size)
            if meters is not None:
                    labels = data.labels.cpu().numpy()
                    if embeddings is not None:
                        embeddings = embeddings.detach().cpu().numpy()
                        for m in meters.metrics:
                            m.update(embeddings, labels, n=self.params.batch_size)
                    iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
            if meters is not None:
                iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
            else:
                iterator.set_postfix({"loss": "{:.2f}".format(losses.avg)})
            if save_every_n is not None:
                if b_idx % save_every_n == 0:
                    self.save_model(self.params.save_path)
        iterator.close()
        if self.verbose and meters is not None:
            meters.display_metrics()
        results = {"loss": losses.avg}
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                m.reset()
        return results




       

        




    









