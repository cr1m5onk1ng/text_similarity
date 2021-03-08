
import os
from pathlib import Path

from onnxruntime.capi.session import Session

from src.training.train import Trainer
from src.models.modeling import BaseEncoderModel
import numpy as np
from src.configurations.config import Configuration
from src.utils.metrics import AverageMeter, Metrics
from ..models.sentence_encoder import SentenceTransformerWrapper, SiameseSentenceEmbedder
import torch
from torch import nn
import torch.nn.functional as F
from ..training.learner import Learner
from ..dataset.dataset import DataLoader, Dataset, SmartParaphraseDataloader
from ..models.losses import SimpleDistillationLoss
from typing import Dict, List
from tqdm import tqdm
from torch.cuda import amp
from queue import PriorityQueue
from heapq import heappush, heappop
from src.models.SentenceTransformer import SentenceTransformer
from sentence_transformers import models, losses, evaluation
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
import onnx
from onnxruntime.quantization import QuantizationMode, quantize, quantize_dynamic

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


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

def prune_rewire(args, model, eval_dataloader, tokenizer, use_tqdm=True):
    results = {}
    model.to(args.device)
    # get the model ffn weights and biases
    inter_weights = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size).to(args.device)
    inter_biases = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size).to(args.device)
    output_weights = torch.zeros(model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size).to(args.device)

    if isinstance(model, SentenceTransformerWrapper):
        layers = model.context_embedder.auto_model.base_model.encoder.layer
    else:
        layers = model.base_model.encoder.layer
    head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(args.device)
    ffn_importance = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size).to(args.device)
    for layer_num in range(model.config.num_hidden_layers):
        inter_weights[layer_num] = layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(args.device)
        inter_biases[layer_num] = layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(args.device)
        output_weights[layer_num] = layers._modules[str(layer_num)].output.dense.weight.detach().to(args.device)

    head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    tot_tokens = 0.0
    for batch in eval_dataloader:
        model.eval()
        batch.to(args.device)
        if args.mixed_precision:
            with amp.autocast():
                if not isinstance(model, SentenceTransformerWrapper):
                    outputs = model(output_attentions=True, **batch.embeddings_features.to_dict(), head_mask=head_mask, labels=batch.labels)
                    tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(features=batch, output_attentions=True, head_mask=head_mask)
                    tmp_eval_loss = outputs.loss
                    logits = outputs.predictions
        else:
            if not isinstance(model, SentenceTransformerWrapper):
                outputs = model(output_attentions=True, **batch.embeddings_features.to_dict(), head_mask=head_mask, labels=batch.labels)
                tmp_eval_loss, logits = outputs[:2]
            else:
                outputs = model(features=batch, output_attentions=True, head_mask=head_mask)
                tmp_eval_loss = outputs.loss
                logits = outputs.predictions

        eval_loss += tmp_eval_loss.mean().item()

        # TODO accumulate? absolute value sum?
        tmp_eval_loss.backward()

        # collect attention confidence scores
        head_importance += head_mask.grad.abs().detach()

        # collect gradients of linear layers
        for layer_num in range(model.config.num_hidden_layers):
            ffn_importance[layer_num] += torch.abs(
                torch.sum(layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach()*inter_weights[layer_num], 1) 
                + layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach()*inter_biases[layer_num])

        tot_tokens += (batch.sentence_1_features["attention_mask"].float().detach().sum().data + batch.sentence_2_features["attention_mask"].float().detach().sum().data )

        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()

    head_importance /= tot_tokens

    # Layerwise importance normalization
    exponent = 2
    norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
    head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # rewire the network
    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    num_heads = model.config.num_attention_heads
    head_size = model.config.hidden_size / num_heads
    for layer_num in range(model.config.num_hidden_layers):
        # load query, key, value weights
        query_weight = layers._modules[str(layer_num)].attention.self.query.weight
        query_bias = layers._modules[str(layer_num)].attention.self.query.bias
        key_weight = layers._modules[str(layer_num)].attention.self.key.weight
        key_bias = layers._modules[str(layer_num)].attention.self.key.bias
        value_weight = layers._modules[str(layer_num)].attention.self.value.weight
        value_bias = layers._modules[str(layer_num)].attention.self.value.bias

        # sort query, key, value based on the confidence scores
        query_weight, query_bias = sort_by_importance(query_weight,
            query_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
        layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = sort_by_importance(key_weight,
            key_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
        layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = sort_by_importance(value_weight,
            value_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
        layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

        # output matrix
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
    path = args.output_dir + "/pruned_" + args.config_name + "_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim))
    Path(path).mkdir(exist_ok=True)

    model.config.hidden_act = 'relu'    # use ReLU activation for the pruned models.
    model.config.num_attention_heads = min([num_heads, args.target_num_heads])
    model.config.intermediate_size = layers._modules['0'].intermediate.dense.weight.size(0)
    model.config.save_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))
    model.save_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))
    tokenizer.save_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))

    return results, preds

def quantize_model(path, save_path):
    #onnx_model = onnx.load(path)
    Path(path)
    Path(save_path)

    quantize_dynamic(
        model_input=path,
        model_output=save_path
    )

def convert_to_onnx(model: nn.Module, params: Configuration, quantize=False):
    print(f"################## Staring ONNX graph optimization on model: {params.model} ##################")
    tokens = params.tokenizer.encode_plus("This is a sample input.")
    model.to(torch.device("cpu"))
    model.eval()

    input_names = ['input_ids', 'attention_mask']
    output_names = ['output_0']

    dynamic_axes = {
        'attention_mask': {
            0: 'batch',
            1: 'sequence'
        },
        'input_ids': {
            0: 'batch',
            1: 'sequence'
        },
        'output_0': {
            0: 'batch',
            1: 'sequence'
        }
    }

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
        opset_version=11,
    )
    print(f"Optimized model correctly exported in: {onnx_path}")
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    print("Checking ONNX model loaded from: {}".format(onnx_path))
    try:
        onnx_options = SessionOptions()
        sess = InferenceSession(onnx_path, onnx_options)
        print("Model loaded successfully")
        output_onnx = sess.run(None, {'input_ids': [tokens['input_ids']],
                                'attention_mask': [tokens['attention_mask']]})
        print(output_onnx)
    except RuntimeException as re:
        print("Error while loading the model: {}".format(re))
    
    if quantize:
        quant_path = save_path + "/quantized_model.onnx"
        print(f"Starting 8 bit quantization on model: {params.model_parameters.model_name}")
        quantize_model(onnx_path, quant_path)
        print(f"Quantized model correctly exported in {quant_path}")

class DistillationStrategy(Learner):
    def __init__(self, teacher: nn.Module, train_dataloader, *args, evaluator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.train_dataloader = train_dataloader
        self.evaluator = evaluator

    def _distillation_step(self, save_every_n=None):
        raise NotImplementedError()

    def distill(self, save_every_n=None, reduce_sentences=None):
        if (self.model.get_sentence_embedding_dimension() < 
            self.teacher.get_sentence_embedding_dimension()):
            assert reduce_sentences is not None
            DistillationStrategy.reduce_teacher_dim(reduce_sentences, self.model, self.teacher)

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
    def reduce_student_dim(student, sentences, dim=128):
        embeddings = student.encode_text(sentences, convert_to_numpy=True)
        pca = PCA(n_components=dim)
        pca.fit(embeddings)
        components = np.asarray(pca.components_)
        projection = nn.Linear(student.get_sentence_embedding_dimension(), dim, bias=False)
        projection.linear.weight = torch.nn.Parameter(torch.tensor(components))
        student.add_module('projection', projection)

    @staticmethod
    def reduce_teacher_dim(sentences, student, teacher):
        logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
        embeddings = teacher.encode(sentences, convert_to_numpy=True)
        pca = PCA(n_components=student.get_sentence_embedding_dimension())
        pca.fit(embeddings)

        #Add projection layer to teacher that projects the embeddings down to the student embedding size
        projection = models.projection(in_features=teacher.get_sentence_embedding_dimension(), out_features=student.get_sentence_embedding_dimension(), bias=False, activation_function=torch.nn.Identity())
        projection.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        teacher.add_module('projection', projection)


class FastFormersDistiller(DistillationStrategy):
    def __init__(self, state_loss_ratio, tr_att_loss_ratio, use_cosine_sim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_loss_ratio = state_loss_ratio
        self.tr_att_loss_ratio = tr_att_loss_ratio
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
            pooled_teacher, outputs_teacher = self.teacher.encode(data.sentence_1_features, output_attention=True)
        pooled_student, outputs_student = self.model.encode(data.sentence_2_features, output_attention=True)
        loss = self._calculate_losses(outputs_teacher, outputs_student)
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (b_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
        return loss, torch.stack([pooled_teacher, pooled_student], dim = 0)

    def _mixed_precision_step(self, data, b_idx):
        with torch.no_grad():
            pooled_teacher, outputs_teacher = self.teacher.encode(data.sentence_1_features, output_attention=True)
        pooled_student, outputs_student = self.model.encode(data.sentence_2_features, output_attention=True)
        loss = self._calculate_losses(outputs_teacher, outputs_student)
        scale_before_step = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss, torch.stack([pooled_teacher, pooled_student], dim=0), scale_before_step

    def _logits_distillation(self, outputs_teacher, outputs_student):
        kd_loss = self.soft_cross_entropy(outputs_student[1], outputs_teacher[1])
        loss = kd_loss
        self.cls_loss += kd_loss
        self.tr_cls_loss.update(kd_loss.item(), n=self.train_dataloader.get_batch_size)
        return loss

    def _embeddings_distillation(self, outputs_teacher, outputs_student):
        teacher_reps = outputs_teacher[2]
        student_reps = outputs_student[2]
        new_teacher_reps = [teacher_reps[0], teacher_reps[self.teacher_layer_num]]
        new_student_reps = [student_reps[0], student_reps[self.model_layer_num]]
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
        self.att_loss += tmp_loss
        self.tr_att_loss.update(tmp_loss.item(), n=self.train_dataloader.get_batch_size)
        return self.tr_att_loss_ratio * self.att_loss

    def _calculate_losses(self, outputs_teacher, outputs_student):
        loss = self._logits_distillation(outputs_teacher, outputs_student)
        loss += self._embeddings_distillation(outputs_teacher, outputs_student)
        loss += self._attention_distillation(outputs_teacher, outputs_student)
        return loss
    
    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).sum(dim=-1).mean() 

    def _distillation_step(self, save_every_n=None):
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
            data.to_device(self.params.device) 
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


class SentenceEncoderDistiller(DistillationStrategy):
    """
    Distiller module based on SBERT implementation
    """
    def __init__(
        self, 
        layers,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
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
        iterator = tqdm(self.train_dataloader, total=len(self.train_dataloader))
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

       




    









