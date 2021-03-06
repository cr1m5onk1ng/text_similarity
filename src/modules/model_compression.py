
import os

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

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


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
            model = self.model.context_embedder.auto_model
        else:
            model = self.model
        if layers is not None:
            if "distilbert" in self.params.model:
                layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.transformer.layer) if i in layers])
                model.transformer.layer = layers_to_keep
                model.config.n_layers = len(layers_to_keep)
                self.model.context_embedder.auto_model = model
            else:
                layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.encoder.layer) if i in layers])
                model.encoder.layer = layers_to_keep
                model.config.num_hidden_layers = len(layers_to_keep)
                self.model.context_embedder.auto_model = model
            assert self.model.context_embedder.auto_model.config.num_hidden_layers == len(layers)

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

       




    









