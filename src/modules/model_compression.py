
import os
from sys import setdlopenflags

from transformers.utils.dummy_tf_objects import TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST
from src.modules.pooling import AvgPoolingStrategy, EmbeddingsPooler, EmbeddingsSimilarityCombineStrategy, SentenceBertCombineStrategy

import sentence_transformers
import transformers

from src.training.train import Trainer
from src.models.modeling import BaseEncoderModel
import numpy as np
from src.configurations.config import Configuration
from src.utils.metrics import AverageMeter, Metrics
from ..models.sentence_encoder import SiameseSentenceEmbedder
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
from sentence_transformers import SentenceTransformer, models, losses, evaluation
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class Pruner(Learner):
    """
    Pruner model based on structured pruning
    of the least importart attention heads and intermediate layers
    based on Fastformers approach

    params:
        model: the model to prune
        n_heads: the number of least important heads tu prune
        n_dense: the number of the least important intermediate dense layers to prune
    """
    def __init__(
        self, 
        output_path: str, 
        n_heads: int, 
        n_dense: int,
        *args,
        **kwargs
    ):
        # get the model intermediate layer weights and biases
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.n_dense = n_dense
        self.output_path = output_path
        if isinstance(self.model, SentenceTransformer):
            model = self.model._first_module().auto_model
        elif isinstance(self.model, SiameseSentenceEmbedder):
            model = self.model.context_embedder
        else:
            model = self.model
        self.inter_weights = torch.zeros(
            model.config.num_hidden_layers, 
            model.config.intermediate_size,
            model.config.hidden_size).to(self.params.device)
        self.inter_biases = torch.zeros(
            model.config.num_hidden_layers,
            model.config.hidden_size).to(self.params.device)
        self.output_weights = torch.zeros(
            model.config.num_hidden_layers,
            model.config.hidden_size,
            model.config.intermediate_size).to(self.params.device)

        #a list of all the layers of the transformer model
        self.layers = model.base_model.encoder.layer

        self.head_importance = torch.zeros(
            self.model.config.num_hidden_layers,
            self.model.config.num_attention_heads).to(self.params.device)
        self.ffn_importance = torch.zeros(
            self.model.config.num_hidden_layers,
            self.model.config.intermediate_size).to(self.params.device)

        # filling the weights with the original data
        for layer_num in range(model.config.num_hidden_layers):
            self.inter_weights[layer_num] = self.layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(self.params.device)
            self.inter_biases[layer_num] = self.layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(self.params.device)
            self.output_weights[layer_num] = self.layers._modules[str(layer_num)].output.dense.weight.detach().to(self.params.device)

        #the parameters to optimize
        self.head_mask = torch.ones(
            model.config.num_hidden_layers,
            model.config.num_attention_heads, 
            requires_grad=True).to(self.params.device)

    def update_importance(self):
        for layer_num in range(self.model.context_edbedder.model.config.num_hidden_layers):
            self.head_importance += self.head_mask.grad.abs().detach()
            self.ffn_importance[layer_num] += torch.abs(
                torch.sum(self.layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach() * self.inter_weights[layer_num], 1)
                + self.layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach() * self.inter_biases[layer_num])

    def layer_importance_normalization(self):
        exponent = 2
        # L2 norm
        norm_by_layer = torch.pow(torch.pow(self.head_importance, exponent).sum(-1), 1 / exponent)
        self.head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20   

    def sort_by_importance(self, weight, bias, importance, num_instances, stride):
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

    def rewire(self):
        head_importance = self.head_importance.cpu()
        ffn_importance = self.ffn_importance.cpu()
        num_heads = self.model.config.num_attention_heads
        head_size = self.model.config.hidden_size / num_heads

        for layer_num in range(self.model.config.num_hidden_layers):
            query_weight = self.layers._modules()[str(layer_num)].attention.self.query.weight
            query_bias = self.layers._modules[str(layer_num)].attention.self.query.bias
            key_weight = self.layers._modules[str(layer_num)].attention.self.key.weight
            key_bias = self.layers._modules[str(layer_num)].attention.self.key.bias
            value_weight = self.layers._modules[str(layer_num)].attention.self.value.weight
            value_bias = self.layers._modules[str(layer_num)].attention.self.value.bias

            # sort query, key and value matrices by confidence scores
            heads_to_prune = self.n_heads

            query_weight, query_bias = self.sort_by_importance(
                query_weight,
                query_bias,
                head_importance[layer_num],
                heads_to_prune,
                head_size
            )
            self.layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
            self.layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)

            value_weight, value_bias = self.sort_by_importance(
                value_weight,
                value_bias,
                head_importance[layer_num],
                heads_to_prune,
                head_size
            )
            self.layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
            self.layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

            key_weight, key_bias = self.sort_by_importance(
                key_weight,
                key_bias,
                head_importance[layer_num],
                heads_to_prune,
                head_size
            )
            self.layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
            self.layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)

            #Sort attention outputs
            weight_sorted, _ = self.sort_by_importance(
                self.layers._modules[str(layer_num)].attention.output.dense.weight.transpose(0, 1),
                None,
                head_importance[layer_num],
                heads_to_prune,
                head_size
            )
            self.layers._modules[str(layer_num)].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)

            #Sort intermediate
            ffn_to_prune = self.n_dense
            weight_sorted, bias_sorted = self.sort_by_importance(
                self.layers._modules[str(layer_num)].intermediate.dense.weight,
                self.layers._modules[str(layer_num)].intermediate.dense.bias, 
                ffn_importance[layer_num],
                ffn_to_prune,
                1)
            self.layers._modules[str(layer_num)].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
            self.layers._modules[str(layer_num)].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)

            #Sort output
            weight_sorted, _ = self.sort_by_importance(
                self.layers._modules[str(layer_num)].output.dense.weight.transpose(0, 1),
                None, 
                ffn_importance[layer_num],
                ffn_to_prune,
                1)
            weight_sorted = weight_sorted.transpose(0, 1)
            self.layers._modules[str(layer_num)].output.dense.weight = torch.nn.Parameter(weight_sorted)
        self.model.config.hidden_act = 'relu'
        self.model.config.num_attention_heads = min([num_heads, self.n_heads])
        self.model.config.intermediate_size = self.layers._modules['0'].intermediate.dense.weight.size(0)


    def prune(self, data_loader):
        logging.info(f"##### Pruning some good sambuco from model: {self.config_name} #####")
        total_tokens = 0.
        losses = AverageMeter("loss")
        if self.metrics is not None:
            meters = Metrics(*self.metrics["validation"], mode="validation", return_predictions=False)
        else:
            meters = None
        self.model.to(self.params.device)
        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(iterator):
                 ### EVAL LOGIC ###
                data.to_device(self.params.device)
                if self.fp16:
                    with amp.autocast():
                        #pass head mask to the model
                        model_output = self.model(data, head_mask = self.head_mask)
                else:
                    model_output = self.model(data)
                loss = model_output.loss
                logits = model_output.predictions
                if self.use_mean_loss:
                    loss = loss.mean()
                losses.update(loss.item(), self.params.batch_size)

                if meters is not None:
                    labels = data.labels.cpu().numpy()
                    if logits is not None:
                        logits = logits.detach().cpu().numpy()
                        for m in meters.metrics:
                            m.update(logits, labels, n=data_loader.get_batch_size)
                    iterator.set_postfix(loss=losses.avg, **meters.set_postfix())

                ### PRUNING LOGIC ###

                #Updating the importance of heads and ffn
                #by multiplying their weights by the absolute value
                #of the mask gradients
                self.update_importance()
                total_tokens += data.attention_mask.float().detach().sum().data
            iterator.close()
        
        if self.verbose and meters is not None:
            meters.display_metrics()
        results = {"loss": losses.avg}
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                m.reset()

        self.head_importance /= total_tokens

        #l2 normalization of the layers importance
        #TODO make this optional
        self.layer_importance_normalization()

        #Rewiring the network
        self.rewire()

        #Saving the model
        self.save_model(self.output_path)
       
        return results

class Distiller(Learner):
    def __init__(
        self, 
        teacher_model: nn.Module, 
        train_dataloader: DataLoader,
        model_save_path: str,
        *args,
        evaluator=None,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.train_dataloader = train_dataloader
        self.model_save_path = model_save_path
        self.evaluator = evaluator

    def _reduce_dim(self, sentences):
        logging.info("Student model has fewer dimensions that the teacher. Compute PCA for down projection")
        pca_embeddings = self.teacher.encode(sentences, convert_to_numpy=True)
        pca = PCA(n_components=self.model_model.get_sentence_embedding_dimension())
        pca.fit(pca_embeddings)

        #Add Dense layer to teacher that projects the embeddings down to the student embedding size
        dense = models.Dense(in_features=self.teacher.get_sentence_embedding_dimension(), out_features=self.model_model.get_sentence_embedding_dimension(), bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        self.teacher.add_module('dense', dense)


class SentenceTransformersDistiller():
    def __init__(self, params, student_model, teacher_model, train_dataset, evaluators, model_save_path, *args, layers = (1, 4, 7, 10), **kwargs):
        self.params = params
        self.model = student_model
        self.teacher = teacher_model
        self.train_dataset = train_dataset
        self.evaluators = evaluators
        self.model_save_path = model_save_path
        self.layers = layers
        model = self.model._first_module().auto_model
        if layers is not None:
            if "distilbert" in self.params.model:
                layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.encoder.layer) if i in layers])
                model.encoder.layer = layers_to_keep
                model.config.num_hidden_layers = len(layers)
                self.model._first_module().auto_model = model
            else:
                layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.layer) if i in layers])
                model.layer = layers_to_keep
                model.config.num_hidden_layers = len(layers)
                self.model._first_module().auto_model = model
            assert len(self.model._first_module().auto_model.layer) == len(args.layers)
        self.loss = losses.MSELoss(model=self.model)

    def distill(self, reduce_sentences=None):
        logging.info("######## Loading parallel sentences ########")


        logging.info("######## Starting model distillation ########")
        if (self.model_model.get_sentence_embedding_dimension() < 
            self.teacher.get_sentence_embedding_dimension()):
            if reduce_sentences is not None:
                self._reduce_dim(reduce_sentences)

        self.model_model.fit(
            train_objectives=[(self.train_dataloader, self.loss)],
            evaluator=evaluation.SequentialEvaluator(
                self.evaluators,
                main_score_function=lambda scores: np.mean(scores)),
                epochs=self.params.epochs,
                warmup_steps = self.params.warmup_steps,
                output_path=self.model_save_path,
                save_best_model=True,
                optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
                use_amp=True
        ) 
        

class DistillationStrategy(Learner):
    def __init__(self, teacher: nn.Module, train_dataloader, *args, evaluator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.train_dataloader = train_dataloader
        self.evaluator = evaluator

    def _distillation_step(self, save_every_n=None):
        raise NotImplementedError()

    def distill(self, save_every_n=None):
        if self.fp16:
            print("#### Using mixed precision training ####")
            print()
        min_loss = np.inf
        for epoch in range(self.params.epochs):
            print(f"###### EPOCH {epoch+1} #######")
            results = self._distillation_step(save_every_n=save_every_n)
            loss = results["loss"]
            if loss < min_loss:
                min_loss = loss
                self.save_model(os.path.join(self.params.save_path, self.config_name))
            if self.evaluator is not None:
                self.evaluator.evaluate()

    def reduce_dim(self, sentences):
        logging.info("Student model has fewer dimensions that the teacher. Compute PCA for down projection")
        pca_embeddings = self.teacher.encode(sentences, convert_to_numpy=True)
        pca = PCA(n_components=self.model.context_embedder.config.hidden_size)
        pca.fit(pca_embeddings)

        #Add Dense layer to teacher that projects the embeddings down to the student embedding size
        dense = models.Dense(in_features=self.teacher.get_sentence_embedding_dimension(), out_features=self.model_model.get_sentence_embedding_dimension(), bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        self.teacher.add_module('dense', dense)


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
        self.teacher_layer_num = self.teacher.context_embedder.config.num_hidden_layers
        self.model_layer_num = self.model.context_embedder.config.num_hidden_layers

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
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()
            skip_scheduler = False
            if self.fp16:
                loss, embeddings, scale_before_step = self._mixed_precision_step(data, b_idx)
                skip_scheduler = self.scaler.get_scale() != scale_before_step
            else:
               loss, embeddings = self._step(data, b_idx)

            if b_idx > 0:
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
        *args,
        layers=[1, 4, 7, 10],
        multilingual=False,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        #self.loss = nn.MSELoss()
        self.multilingual = multilingual
        if isinstance(self.model, SentenceTransformer):
            model = self.model._first_module().auto_model
        elif isinstance(self.model, SiameseSentenceEmbedder):
            model = self.model.context_embedder
        else:
            model = self.model
        if layers is not None:
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
            assert self.model.context_embedder.config.num_hidden_layers == len(layers)

    def _mse_loss_multi(self, teacher_embeddings, student_embeddings_src, student_embeddings_tgt):
        src_square_diff = F.mse_loss(student_embeddings_src, teacher_embeddings)
        tgt_square_diff = F.mse_loss(student_embeddings_tgt, teacher_embeddings)
        return src_square_diff + tgt_square_diff

    def _mse_loss(self, student_embeddings, teacher_embeddings):
        return F.mse_loss(student_embeddings, teacher_embeddings)

    def _step(self, data, b_idx):
        with torch.no_grad():
            if isinstance(self.teacher, SentenceTransformer):
                    if not self.multilingual:
                        embeddings_1 = self.teacher.encode(data.sentences, convert_to_tensor=True, device=self.params.device, show_progress_bar=False)
                    else:
                        embeddings_1 = self.teacher.encode(data.src_sentences, convert_to_tensor=True, device=self.params.device, show_progress_bar=False)
                    embeddings_1 = embeddings_1.to(self.params.device)
            else:
                if not self.multilingual:
                    embeddings_1 = self.teacher.encode(data.features)
                else:
                    embeddings_1 = self.teacher.encode(data.setence_1_features)

        if not self.multilingual:
                embeddings_2 = self.model.encode(data.features)
        else:
            embeddings_2_src = self.model.encode(data.sentence_1_features)
            embeddings_2_tgt = self.mode.encode(data.sentence_2_features)       

        if not self.multilingual:
            loss = self._mse_loss(student_embeddings=embeddings_2, teacher_embeddings=embeddings_1)  
        else:
            loss = self._mse_loss_multi(embeddings_1, embeddings_2_src, embeddings_2_tgt)
            
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (b_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
        return loss, torch.stack([embeddings_1, embeddings_2_tgt], dim=0)

    def _mixed_precision_step(self, data, b_idx):
        with amp.autocast():
            with torch.no_grad():
                if isinstance(self.teacher, SentenceTransformer):
                    if not self.multilingual:
                        embeddings_1 = self.teacher.encode(data.sentences, convert_to_tensor=True, device=self.params.device, show_progress_bar=False)
                    else:
                        embeddings_1 = self.teacher.encode(data.src_sentences, convert_to_tensor=True, device=self.params.device, show_progress_bar=False)
                    embeddings_1 = embeddings_1.to(self.params.device)
                else:
                    if not self.multilingual:
                        embeddings_1 = self.teacher.encode(data.features)
                    else:
                        embeddings_1 = self.teacher.encode(data.setence_1_features)
            if not self.multilingual:
                embeddings_2 = self.model.encode(data.features)
            else:
                embeddings_2_src = self.model.encode(data.sentence_1_features)
                embeddings_2_tgt = self.mode.encode(data.sentence_2_features)
        if not self.multilingual:
            loss = self._mse_loss(student_embeddings=embeddings_2, teacher_embeddings=embeddings_1)  
        else:
            loss = self._mse_loss_multi(embeddings_1, embeddings_2_src, embeddings_2_tgt)
        scale_before_step = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
  
        return loss, torch.stack([embeddings_1, embeddings_2 if not self.multilingual else embeddings_2_tgt], dim=0), scale_before_step

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
            data.to_device(self.params.device)
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

       




    









