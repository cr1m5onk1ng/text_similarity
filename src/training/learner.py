import os
from src.modules.modules import ClassifierOutput
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.configurations.config import Configuration
import torch
from tqdm import tqdm
from torch.cuda import amp
from src.utils.metrics import AverageMeter, AccuracyMeter, Metrics
from src.models.modeling import BaseEncoderModel, TransformerWrapper
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Union, Dict, List
import numpy as np


class Learner:
    def __init__(
        self,
        params: Configuration,
        config_name: str,
        model: torch.nn.Module,
        steps: int = 0,
        accumulation_steps: int = 1,
        warm_up_steps: int = 0,
        fp16: bool = True,
        use_tpu = False,
        max_grad_norm: int = 1,
        use_mean_loss: bool = False,
        metrics: Union[Dict[str, List[AverageMeter]], None] = None,
        verbose: bool = True,
        replacing_rate_scheduler = None,
    ):
        self.params = params
        self.config_name = config_name
        self.model = model
        self.steps = steps
        self.warm_up_steps = warm_up_steps
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.use_tpu = use_tpu
        self.use_mean_loss = use_mean_loss
        self.metrics = metrics
        self.scaler = None
        self.verbose = verbose
        if self.use_tpu:
            self.fp16 = False
        if self.fp16:
            self.scaler = amp.GradScaler() 
        self.optimizer = Learner.set_up_optimizer(self.model, self.params.lr)
        self.scheduler = Learner.set_up_scheduler(self.optimizer, self.steps, self.warm_up_steps)
        self.replacing_rate_scheduler = replacing_rate_scheduler
    @staticmethod
    def set_up_optimizer(model, lr):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
            "weight_decay": 0.01,
            },
            {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_parameters, lr=lr, eps=1e-6) 

    @staticmethod
    def set_up_scheduler(optimizer, steps, warm_up_steps):
        scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=steps
        )
        return scheduler 

    def save_model(self, path):
        #Saving model
        assert(path is not None)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        d = {}

        if hasattr(self.model, 'context_embedder'):
            self.model.context_embedder.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        #d['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(d, os.path.join(path, f"{self.config_name}_optimizer_state.bin"))

        #Saving parameters
        torch.save(self.params, os.path.join(path, "training_params.bin"))
    
    def _reduce_fn(vals):
        # take average
        return sum(vals) / len(vals)

    def step(self, data, b_idx):
        if isinstance(self.model, BaseEncoderModel):
            if self.model.input_dict:
                model_output = self.model(**data.to_dict())
            else:
                model_output = self.model(data)
            loss = model_output.loss
            if hasattr(model_output, "predictions"):
                logits = model_output.predictions
        else:
            if not isinstance(data, dict):
                features = data.to_dict()
                labels = data.labels
            else:
                features = data
                labels = data["labels"]
            model_output = self.model(**features, labels=labels)
            loss = model_output[0]
            logits = model_output[1]
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (b_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
        return loss, logits

    def mixed_precision_step(self, data, b_idx):
        logits = None
        with amp.autocast():
            if isinstance(self.model, BaseEncoderModel):
                if self.model.input_dict:
                    model_output = self.model(**data.to_dict())
                else:
                    model_output = self.model(data)
                loss = model_output.loss
                if hasattr(model_output, "predictions"):
                    logits = model_output.predictions
            else:
                if not isinstance(data, dict):
                    features = data.to_dict()
                    labels = data.labels
                else:
                    features = data
                    labels = data["labels"]
                model_output = self.model(**features, labels=labels)
                loss = model_output[0]
                logits = model_output[1]
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        scale_before_step = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
  
        return loss, logits, scale_before_step

    def _tpu_step(self, data, b_idx, multi_core=False):
        model_output = self.model(data)
        loss = model_output.loss
        logits = None
        if hasattr(model_output, "predictions"):
            logits = model_output.predictions
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (b_idx + 1) % self.accumulation_steps == 0:
            xm.optimizer_step(self.optimizer, barrier = not multi_core)
        return loss, logits


    def to_device(self, data, device):
        if isinstance(data, dict):
            for k in data:
               if isinstance(data[k], torch.Tensor):
                   data[k] = data[k].to(device)   
        else:
            data.to(device) 
        
      
    def train_fn(self, data_loader):
        if self.use_tpu:
            try:
                import torch_xla 
                import torch_xla.core.xla_model as xm
            except:
                ImportError("Pytorch XLA is not available")
            self.device = xm.xla_device()
        losses = AverageMeter("loss")
        if self.metrics is not None:
            meters = Metrics(*self.metrics["training"])
        else:
            meters = None
        iterator = tqdm(data_loader, total=len(data_loader))
        self.model.to(self.params.device)
        self.model.train()
        results = []
        for b_idx, data in enumerate(iterator):
            self.to_device(data, self.params.device)
            skip_scheduler = False
            if self.use_tpu:
                loss, logits = self._tpu_step(data, b_idx)
            elif self.fp16:
                loss, logits, scale_before_step = self.mixed_precision_step(data, b_idx)
                skip_scheduler = self.scaler.get_scale() != scale_before_step
            else:
               loss, logits = self.step(data, b_idx)

            if (b_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                if not skip_scheduler:
                    self.scheduler.step()
            if self.replacing_rate_scheduler is not None:
                self.replacing_rate_scheduler.step()
        
            losses.update(loss.item(), self.params.batch_size)
            if meters is not None:
                if isinstance(data, dict):
                    labels = data["labels"].cpu().numpy()
                else:
                    labels = data.labels.cpu().numpy()
                if logits is not None:
                    logits = logits.detach().cpu().numpy()
                    for m in meters.metrics:
                        m.update(logits, labels, n=self.params.batch_size)
                if not self.use_tpu:
                    iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
            if not self.use_tpu:
                if meters is not None:
                    iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
                else:
                    iterator.set_postfix({"loss": "{:.2f}".format(losses.avg)})
        if not self.use_tpu:
            iterator.close()
        if self.verbose and meters is not None and not self.use_tpu:
            meters.display_metrics()
        results = {"loss": losses.avg}
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                m.reset()
        return results

    def eval_fn(self, data_loader, return_predictions=False):
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
                self.to_device(data, self.params.device)
                if self.fp16:
                    with amp.autocast():
                        if isinstance(self.model, BaseEncoderModel):
                            if self.model.input_dict:
                                model_output = self.model(**data.to_dict())
                            else:
                                model_output = self.model(data)
                            if isinstance(model_output, ClassifierOutput):
                                loss = model_output.loss
                                logits = model_output.predictions
                            else:
                                logits = model_output
                                loss = None
                        else:
                            if not isinstance(data, dict):
                                features = data.to_dict()
                                labels = data.labels
                            else:
                                features = data
                                labels = data["labels"]
                            model_output = self.model(**features, labels=labels)
                            loss = model_output[0]
                            logits = model_output[1]
                else:
                    if isinstance(self.model, BaseEncoderModel):
                        if self.model.input_dict:
                            model_output = self.model(**data.to_dict())
                        else:
                            model_output = self.model(data)
                        if isinstance(model_output, ClassifierOutput):
                                loss = model_output.loss
                                logits = model_output.predictions
                        else:
                            logits = model_output
                            loss = None
                    else:
                        if not isinstance(data, dict):
                            features = data.to_dict()
                            labels = data.labels
                        else:
                            features = data
                            labels = data["labels"]
                        model_output = self.model(**features, labels=labels)
                        loss = model_output[0]
                        logits = model_output[1]
                if loss is not None:
                    losses.update(loss.item(), self.params.batch_size)
                if meters is not None:
                    if isinstance(data, dict):
                        labels = data["labels"].cpu().numpy()
                    else:
                        labels = data.labels.cpu().numpy()
                    if logits is not None:
                        logits = logits.detach().cpu().numpy()
                        for m in meters.metrics:
                            m.update(logits, labels, n=self.params.batch_size)
                    postfix_dict = meters.set_postfix()
                    if loss is not None:
                        postfix_dict["loss"] = losses.avg
                    iterator.set_postfix(**postfix_dict)
            iterator.close()
        if self.verbose and meters is not None:
            meters.display_metrics()
        results = {"loss": losses.avg}
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                if return_predictions:
                    results[f"predictions_{m.get_name}"] = m.all_predictions
                    results[f"labels_{m.get_name}"] = m.all_labels
                m.reset()
        return results



