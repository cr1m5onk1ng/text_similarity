import numpy as np
import os
import joblib
import torch
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from src.configurations import config as config
from src.dataset.dataset import *
from src.training.learner import Learner
from src.utils.metrics import *
from src.utils import utils
from typing import Union, List, Dict
import logging
import gc
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

class Trainer:
    def __init__(
        self, 
        name: str, 
        train_dataloader: DataLoader, 
        valid_dataloader: DataLoader, 
        epochs: int, 
        configuration: Learner,
        measure: str = "loss", 
        direction: str = "minimize", 
        verbose: bool = True, 
        early_stopping: bool = False,
        return_predictions: bool = False,
        convert_pred_to_numpy: bool = False,
        eval_in_train=False,
        model_save_path: str = "trained_models"
        ):
      
        self.name = name
        self.train_data_loader = train_dataloader
        self.valid_data_loader = valid_dataloader
        self.epochs = epochs
        self.configuration = configuration 
        self.measure = measure
        self.direction = direction
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.return_predictions = return_predictions
        self.convert_pred_to_numpy = convert_pred_to_numpy
        self.eval_in_train=eval_in_train
        self.model_save_path = model_save_path

    def execute(self, write_results=False):
        messages = []
        results = {}
        
        merge_strategy = "None"
        if hasattr(self.configuration.model, "merge_strategy"):
            merge_strategy = self.configuration.model.merge_strategy
        title = f" ------- Running training for model: {self.configuration.model.model_name} with config: {self.configuration.config_name} -------- \n"
        configs = f"Params: batch size: {self.train_data_loader.get_batch_size}; number of epochs: {self.epochs}; frozen weights: { self.configuration.model.params.model_parameters.freeze_weights}; pretrained embeddings: {self.configuration.model.params.model_parameters.use_pretrained_embeddings}"
        configs += f"\nModel params: hidden size: {self.configuration.model.params.model_parameters.hidden_size}; lr: {self.configuration.model.params.lr};"
        configs += f" merge strategy: {merge_strategy}"
        if self.verbose:
            logging.info(title)
            print()
            logging.info(configs)
        messages.append(title)
        messages.append(configs)
        best_metric = np.inf if self.direction == "minimize" else np.NINF
        for epoch in range(self.epochs):
            print()
            logging.info(f"######## Epoch: {epoch+1} #########")
            print()
            train_res = self.configuration.train_fn(self.train_data_loader)
            if self.configuration.eval_in_train:
                valid_res = self.configuration.eval_fn(self.valid_data_loader, return_predictions=self.return_predictions)
            optim_metric = valid_res[self.measure] if self.eval_in_train else train_res[self.measure]
            train_res_line = f"training > epoch: {epoch+1}; "
            for metric, value in train_res.items():
                train_res_line += f"{metric}: {value}; "
            if self.eval_in_train:
                valid_res_line = f"validation > epoch: {epoch+1}; "
                for metric, value in valid_res.items():
                    valid_res_line += f"{metric}: {value}; "
                messages.append(valid_res_line)
            messages.append(train_res_line)
            if self.direction == "minimize":
                if optim_metric < best_metric:
                    best_metric = optim_metric
                    if self.return_predictions:
                      results["labels"] = valid_res[f"labels_{self.measure}"]
                      results["predictions"] = valid_res[f"predictions_{self.measure}"]
                    self.configuration.save_model(os.path.join(self.model_save_path, self.configuration.config_name))
            elif self.direction == "maximize":
                if optim_metric > best_metric:
                    best_metric = optim_metric
                    if self.return_predictions:
                      results["labels"] = valid_res[f"labels_{self.measure}"]
                      results["predictions"] = valid_res[f"predictions_{self.measure}"]
                    self.configuration.save_model(os.path.join(self.model_save_path, self.configuration.config_name))
            results["best_metric"] = best_metric
            
        messages.append("\n")
        if write_results:
            logging.info("writing results to file")
            with open(os.path.join("results", self.name), "w+") as f:
                for m in messages:
                    f.write(m+"\n")
        return results

        
if __name__ == "__main__":
   
    
    
    """
    sim_metrics = {"training": [SimilarityAveragePrecisionMeter, SimilarityAccuracyMeter], "validation": [SimilarityAveragePrecisionMeter, SimilarityAccuracyMeter]}
    soft_metrics = {"training": [AccuracyMeter, F1Meter], "validation": [SimilarityAveragePrecisionMeter, SimilarityAccuracyMeter]}
    model = SiameseSentenceEmbedder(train_model=True, use_sense_embeddings=True, loss="online_contrastive", merge_strategy="combine", pooling_strategy="cls")
    train_data_loader = utils.load_file(f"../dataset/cached/paws/train_en_sense_16")
    valid_data_loader = utils.load_file(f"../dataset/cached/paws/valid_en_sense_16")
    #valid_data_loader = utils.load_file(f"../dataset/cached/pawsx_test_all_languages_{config.BATCH_SIZE}")
    num_train_steps = len(train_data_loader) * config.EPOCHS
    num_warmup_steps = int(num_train_steps*0.1)
    name = "siamese_ocontrastive_combine_cls"
    measure = "loss"
    direction = "minimize"

    learner = Learner(
        config_name=name, 
        model=model, 
        lr=config.LR, 
        bs=config.BATCH_SIZE, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        device=config.DEVICE, 
        fp16=True, 
        metrics=sim_metrics,
        eval_in_train=False
        )

    trainer = Trainer(
        name, 
        train_data_loader, 
        valid_data_loader, 
        config.EPOCHS, 
        configuration=learner,
        direction=direction,
        measure=measure,
        )

    trainer.execute(write_results=True)
    
    """

    """
    soft_metrics = {"training": [AccuracyMeter, F1Meter], "validation": [AccuracyMeter, F1Meter]}

    model = MBERTClassifier(train_model=True, use_sense_embeddings=False, senses_as_features=True, strategy="avg")

    train_data_loader = utils.load_file(f"../dataset/cached/paws/train_pairs_en_16")
    valid_data_loader = utils.load_file(f"../dataset/cached/paws/valid_pairs_en_16")
    num_train_steps = len(train_data_loader) * config.EPOCHS
    num_warmup_steps = int(num_train_steps*0.1)
    name = "mbert_no_sense_avg"
    learner = Learner(
        config_name=name, 
        model=model, 
        lr=config.LR, 
        bs=config.BATCH_SIZE, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        device=config.DEVICE, 
        fp16=True, 
        metrics=soft_metrics,
        eval_in_train=False
    )

    trainer = Trainer(
        name, 
        train_data_loader, 
        valid_data_loader, 
        config.EPOCHS, 
        configuration=learner,
        direction="minimize",
        measure="loss"
    )
    
    trainer.execute(write_results=True)

    """
    
    """
    soft_metrics = {"training": [AccuracyMeter, F1Meter], "validation": [AccuracyMeter, F1Meter]}

    model = MBERTBaseline(train_model=False, use_sense_embeddings=True, senses_as_features=True, strategy="avg")

    train_data_loader = utils.load_file(f"../dataset/cached/paws/train_en_sense_16")
    valid_data_loader = utils.load_file(f"../dataset/cached/paws/valid_en_sense_16")
    num_train_steps = len(train_data_loader) * config.EPOCHS
    num_warmup_steps = int(num_train_steps*0.1)
    name = "mbert_baseline"
    learner = Learner(
        config_name=name, 
        model=model, 
        lr=config.LR, 
        bs=config.BATCH_SIZE, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        device=config.DEVICE, 
        fp16=True, 
        metrics=soft_metrics,
        eval_in_train=False
    )

    trainer = Trainer(
        name, 
        train_data_loader, 
        valid_data_loader, 
        config.EPOCHS, 
        configuration=learner,
        direction="minimize",
        measure="loss"
    )
    
    trainer.execute(write_results=True)
    """
    
    """
    processor = WicProcessor()
    train_dataset = processor.build_dataset("../data/WiC/train/train.data.txt", "../data/WiC/train/train.gold.txt")
    valid_dataset = processor.build_dataset("../data/WiC/dev/dev.data.txt", "../data/WiC/dev/dev.gold.txt")
    train_data_loader = WiCDataLoader.build_batches(train_dataset, 8)
    valid_data_loader = WiCDataLoader.build_batches(valid_dataset, 8)

    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]} 

    model = ParaphraseClassifierModelV2(use_sense_embeddings=True, train_model=False, merge_strategy="double_sense")

    num_train_steps = len(train_data_loader) * config.EPOCHS
    num_warmup_steps = 0

    learner = Learner(
        config_name="train_wic_ares_mono_8epochs", 
        model=model, 
        lr=config.LR, 
        bs=config.BATCH_SIZE, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        device=config.DEVICE, 
        fp16=True, 
        metrics=metrics)

    trainer = Trainer(
        "paws_mbert_sense_trained_5epochs", 
        train_data_loader, valid_data_loader, 
        config.EPOCHS, 
        configuration=learner,
        measure="accuracy",
        direction="maximize", 
        return_predictions=True)

    results = trainer.execute(write_results=True)

    plot_roc_curve(results["labels"], results["predictions"], savepath="graphs/roc_cruve")
    """

   
    
