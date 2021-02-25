from os import name
import os

import sentence_transformers

from src.training.train import Trainer
from src.models.modeling import BaseEncoderModel
import numpy as np
from src.configurations.config import Configuration
from src.utils.metrics import AverageMeter, Metrics
from ..models.sentence_encoder import SiameseSentenceEmbedder
import torch
from torch import nn
from ..training.learner import Learner
from ..dataset.dataset import DataLoader, Dataset, SmartParaphraseDataloader
from ..models.losses import SimpleDistillationLoss
from typing import Dict, List
import tqdm
from torch.cuda import amp
from queue import PriorityQueue
from heapq import heappush, heappop
from sentence_transformers import SentenceTransformer, models, losses, evaluation
from sklearn.decomposition import PCA

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
            self.inter_weights[layer_num] = self.layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(params.device)
            self.inter_biases[layer_num] = self.layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(params.device)
            self.output_weights[layer_num] = self.layers._modules[str(layer_num)].output.dense.weight.detach().to(params.device)

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
            meters = Metrics(*self.metrics["validation"], mode="validation", return_predictions=return_predictions)
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

class Distiller:
    def __init__(
        self, 
        params,
        student_model,
        teacher_model, 
        train_dataloader,
        model_save_path,
        *args,
        **kwargs
        ):
        self.params = params
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_dataloader = train_dataloader
        self.model_save_path = model_save_path


class SentenceTransformersDistiller(Distiller):
    def __init__(self, evaluators, *args, layers = (1, 4, 7, 10), **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluators = evaluators
        self.layers = layers
        model = self.student_model._first_module().auto_model
        layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.encoder.layer) if i in layers])
        model.encoder.layer = layers_to_keep
        model.config.num_hidden_layers = len(layers)
        self.student_model._first_module().auto_model = model
        assert len(self.student_model._first_module().auto_model.layer) == len(args.layers)
        self.loss = losses.MSELoss(model=self.student_model)

    def distill(self, reduce_sentences=None):
        logging.info("######## Starting model distillation ########")
        if (self.student_model.get_sentence_embedding_dimension() < 
            self.teacher_model.get_sentence_embedding_dimension()):
            if reduce_sentences is not None:
                self._reduce_dim(reduce_sentences)

        self.student_model.fit(
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

    def _reduce_dim(self, sentences):
        logging.info("Student model has fewer dimensions that the teacher. Compute PCA for down projection")
        pca_embeddings = self.teacher_model.encode(sentences, convert_to_numpy=True)
        pca = PCA(n_components=self.student_model.get_sentence_embedding_dimension())
        pca.fit(pca_embeddings)

        #Add Dense layer to teacher that projects the embeddings down to the student embedding size
        dense = models.Dense(in_features=self.teacher_model.get_sentence_embedding_dimension(), out_features=self.student_model.get_sentence_embedding_dimension(), bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        self.teacher_model.add_module('dense', dense)
        

class SentenceEncoderDistiller(Learner):
    """
    Distiller module based on SBERT implementation
    """
    def __init__(
        self, 
        teacher_model: SiameseSentenceEmbedder, 
        train_dataloader: DataLoader,
        model_save_path,
        *args,
        layers=(1, 4, 7, 10),
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.train_dataloader = train_dataloader
        self.model_save_path = model_save_path
        model = self.model.context_embedder
        layers_to_keep = nn.ModuleList([l for i, l in enumerate(model.encoder.layer) if i in layers])
        model.encoder.layer = layers_to_keep
        model.config.num_hidden_layers = len(layers_to_keep)
        self.model.context_embedder = model
        

    def distill(self):
        logging.info(f"##### Making some fine liquor with model: {self.learner.config_name}#####")
        best_loss = np.inf
        self.student_model.zero_grad()
        res = self.train_fn(self.train_dataloader)
        self.save_model(self.model_save_path)
        return res["loss"]




    









