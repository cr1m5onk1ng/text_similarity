#from src.configurations.config import Configuration
from src.models.modeling import BaseEncoderModel
from torch.cuda import amp
import torch
from tqdm import tqdm
import numpy as np
#from src.dataset.dataset import DataLoader
from src.utils.metrics import *
from typing import List, Dict, Union, Any



class Evaluator:
    def __init__(
        self,
        params,
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        metrics: Dict[str, List[AverageMeter]],
        fp16: bool = True,
        verbose: bool = True
    ):
        self.params = params
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.metrics = metrics
        self.fp16 = fp16
        self.verbose = verbose

    def to_device(self, data, device):
        if isinstance(data, dict):
            for k in data:
               if isinstance(data[k], torch.Tensor):
                   data[k] = data[k].to(device)   
        else:
            data.to(device) 

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()

    def plot(
        self, 
        labels, 
        predictions, 
        metric, 
        label, 
        title, 
        xlabel, 
        ylabel, 
        savepath, 
        show = False):
        from matplotlib import pyplot as plt

        data_points = metric(labels, predictions)
        plt.plot(*data_points, linestyle='--', label=label)
        # Title
        plt.title(title)
        # Axis labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Show legend
        plt.legend() # 
        plt.savefig(savepath)
        if show:
            plt.show()


class ClassificationEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self):
        if self.metrics is not None:
            meters = Metrics(*self.metrics["validation"], mode="validation", return_predictions=False)
        else:
            meters = None
        self.model.to(self.params.device)
        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(self.data_loader, total=len(self.data_loader))
            for b_idx, data in enumerate(iterator):
                self.to_device(data, self.params.device)
                if self.fp16:
                    with amp.autocast():
                        if isinstance(self.model, BaseEncoderModel):
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
                else:
                    if isinstance(self.model, BaseEncoderModel):
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
                if meters is not None:
                    if isinstance(data, dict):
                        labels = data["labels"].cpu().numpy()
                    else:
                        labels = data.labels.cpu().numpy()
                    if logits is not None:
                        logits = logits.detach().cpu().numpy()
                        for m in meters.metrics:
                            m.update(logits, labels, n=self.params.batch_size)
                    iterator.set_postfix(**meters.set_postfix())
            iterator.close()
        if self.verbose and meters is not None:
            meters.display_metrics()


class ParaphraseEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, threshold: Union[float, None] = None):
        results = {}
        self.model.to(self.params.device)
        self.model.eval()
        if self.metrics is not None:
            meters = Metrics(*self.metrics["validation"], mode="validation", return_predictions=False)
        else:
            meters = None
        with torch.no_grad():
            iterator = tqdm(self.data_loader, total=len(self.data_loader))
            for b_idx, data in enumerate(iterator):
                data.to(self.params.device)
                if self.fp16:
                    with amp.autocast():
                        embeddings_1 = self.model.encode(data.sentence_1_features)
                        embeddings_2 = self.model.encode(data.sentence_2_features)
                else:
                    embeddings_1 = self.model.encode(data.sentence_1_features)
                    embeddings_2 = self.model.encode(data.sentence_2_features)
                embeddings = torch.stack([embeddings_1, embeddings_2], dim=0)
                
                if meters is not None:
                    labels = data.labels.cpu().numpy()
                    if embeddings is not None:
                        embeddings = embeddings.detach().cpu().numpy()
                        for m in meters.metrics:
                            m.update(embeddings, labels, n=self.data_loader.get_batch_size)
                    iterator.set_postfix(**meters.set_postfix())
            iterator.close()
        if self.verbose and meters is not None:
            meters.display_metrics()
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                m.reset()
        return results


class RetrievalEvaluator(Evaluator):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self):
        if self.metrics is not None:
            meters = Metrics(*self.metrics["validation"], mode="validation", return_predictions=self.return_predictions)
        else:
            meters = None
        self.model.to(self.device)
        self.model.eval()
        embeddings = None
        results = {}
        src_sentences = []
        tgt_sentences = []
        src_embeddings = []
        tgt_embeddings = []
        with torch.no_grad():
            iterator = tqdm(self.data_loader, total=len(self.data_loader))
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
                        embeddings = self.model.encode(**data)       
                else:
                    embeddings = self.model.encode(**data)

                sentences_1 = data["sentences_1"]
                sentences_2 = data["sentences_2"]
                src_sentences.extend(sentences_1)
                tgt_sentences.extend(sentences_2)

                src_embeddings.append(embeddings[0])
                tgt_embeddings.append(embeddings[1])
            iterator.close()

        src_embeddings = torch.cat(tuple(src_embeddings), dim=0)
        tgt_embeddings = torch.cat(tuple(tgt_embeddings), dim=0)
        for m in meters.metrics:
            m.update(src_embeddings, tgt_embeddings, src_sentences, tgt_sentences)
        if self.verbose and meters is not None:
            meters.display_metrics()
        if meters is not None:
            for m in meters.metrics:
                results[m.get_name] = m.avg
                if self.return_predictions:
                    results[f"predictions_{m.get_name}"] = m.all_predictions
                    results[f"labels_{m.get_name}"] = m.all_labels
        return results