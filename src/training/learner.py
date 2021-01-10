import torch
from tqdm import tqdm
from torch.cuda import amp
from src.utils.metrics import AverageMeter, AccuracyMeter, Metrics
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Union, Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelOutput:
    loss: Union[torch.Tensor, np.array]


@dataclass
class ClassifierOutput(ModelOutput):
    predictions: Union[torch.Tensor, np.array]


@dataclass
class SimilarityOutput(ModelOutput):
    embeddings: Union[torch.Tensor, np.array, None]
    scores: Union[torch.Tensor, np.array, List[float], None]


class Learner:
    def __init__(
        self,
        config_name: str,
        model: torch.nn.Module,
        lr: float,
        bs: int,
        steps: int,
        device: torch.device,
        accumulation_steps: int = 1,
        warm_up_steps: int = 0,
        fp16: bool = False,
        max_grad_norm: int = 1,
        use_mean_loss: bool = False,
        metrics: Union[List[AverageMeter], None] = None,
        verbose: bool = True,
        eval_in_train: bool = False
    ):
        self.config_name = config_name
        self.model = model
        self.lr = lr
        self.bs = bs
        self.steps = steps
        self.warm_up_steps = warm_up_steps
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.use_mean_loss = use_mean_loss
        self.metrics = metrics
        self.scaler = None
        self.verbose = verbose
        self.eval_in_train = eval_in_train
        if self.fp16:
            self.scaler = amp.GradScaler() 
        self.optimizer = Learner.set_up_optimizer(self.model, self.lr)
        self.scheduler = Learner.set_up_scheduler(self.optimizer, self.steps, self.warm_up_steps)
        
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
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            },
            path
        )

    def train_fn(self, data_loader):
        losses = AverageMeter("loss")
        if self.metrics is not None:
            meters = Metrics(*self.metrics["training"])
        else:
            meters = None
        iterator = tqdm(data_loader, total=len(data_loader))
        self.model.to(self.device)
        self.model.train()
        results = []
        for b_idx, data in enumerate(iterator):
            data.to_device(self.device)
            if self.accumulation_steps == 1 and b_idx == 0:
                self.optimizer.zero_grad()
            skip_scheduler = False
            if self.fp16:
                with amp.autocast():
                    model_output = self.model(data)
                    loss = model_output.loss
                    logits = model_output.predictions
                scale_before_step = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if (b_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                skip_scheduler = self.scaler.get_scale() != scale_before_step
            else:
                model_output = self.model(data)
                loss = model_output.loss
                logits = model_output.predictions
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if (b_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()

            if b_idx > 0:
                self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                if not skip_scheduler:
                    self.scheduler.step()
        
            labels = data.labels.cpu().numpy()
            if logits is not None:
                logits = logits.detach().cpu().numpy()
                if meters is not None:
                    for m in meters.metrics:
                        m.update(logits, labels, n=data_loader.get_batch_size)
            losses.update(loss.item(), data_loader.get_batch_size)
            if meters is not None:
                iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
        iterator.close()
        if self.verbose and meters is not None:
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
        self.model.to(self.device)
        self.model.eval()
        final_predictions = []
        all_labels = []
        with torch.no_grad():
            iterator = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(iterator):
                data.to_device(self.device)
                if self.fp16:
                    with amp.autocast():
                        model_output = self.model(data)
                else:
                    model_output = self.model(data)
                loss = model_output.loss
                logits = model_output.predictions
                
                if self.use_mean_loss:
                    loss = loss.mean()
                losses.update(loss.item(), data_loader.batch_size)
                labels = data.labels.cpu().numpy()
                if logits is not None:
                    logits = logits.detach().cpu().numpy()
                    if meters is not None:
                        for m in meters.metrics:
                            m.update(logits, labels, n=data_loader.get_batch_size)
                if meters is not None:
                    iterator.set_postfix(loss=losses.avg, **meters.set_postfix())
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

    def evaluate(self, data_loader, return_predictions=False, convert_to_numpy=False):
        if self.metrics is not None:
            meters = Metrics(*self.metrics["validation"], mode="validation", return_predictions=return_predictions)
        else:
            meters = None
        self.model.to(self.device)
        self.model.eval()
        final_predictions = []
        all_labels = []
        results = {}
        with torch.no_grad():
            iterator = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(iterator):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)
                    if isinstance(v, dict):
                        for k1, v1 in v.items():
                            if isinstance(v1, torch.Tensor):
                                v[k1] = v1.to(self.device)
                        data[k] =  v
                if self.fp16:
                    with amp.autocast():
                        logits = self.model.encode(**data)
                else:
                    logits = self.model.encode(**data)
                labels = data["labels"].cpu().numpy()
                assert logits is not None, "model is returning None, fix it ya jerk"
                if logits is not None:
                    logits = logits.detach().cpu()
                    if convert_to_numpy:
                        logits = logits.numpy()
                    if meters is not None:
                        for m in meters.metrics:
                            m.update(logits, labels, n=data_loader.get_batch_size)
                if meters is not None:
                    iterator.set_postfix(**meters.set_postfix())
            iterator.close()
        if self.verbose and meters is not None:
            meters.display_metrics()
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                if return_predictions:
                    results[f"predictions_{m.get_name}"] = m.all_predictions
                    results[f"labels_{m.get_name}"] = m.all_labels
        return results


