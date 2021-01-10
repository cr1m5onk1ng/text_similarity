from src.training.train import Trainer
from src.training.learner import Learner
from src.dataset.dataset import *
from src.configurations import classifier_config as config
from src.models.modeling import ParaphraseClassifierModel, ParaphraseClassifierModelV2, SiameseSentenceEmbedder, MBERTClassifier
from src.utils.metrics import AccuracyMeter, SimilarityAccuracyMeter, SimilarityAveragePrecisionMeter, plot_roc_curve
from src.utils import utils as utils
import os
import optuna

class ParamOptimizer:

    def __init__(self, model, test_name, train_data, valid_data, measure="loss", direction="minimize", metrics=None, verbose=True):
        self.model = model
        self.measure = measure
        self.direction = direction
        self.test_name = test_name
        self.train_data = train_data
        self.valid_data = valid_data
        self.metrics = metrics
        self.verbose = verbose

    def _build_params(self, warmup_perc, create_plot=False, **params):
        def objective(trial):
            d = {}
            for param, interval in params.items():
                if param == 'lr':
                    d[param] = trial.suggest_loguniform(param, *interval)
                elif param == 'num_layers':
                    d[param] = trial.suggest_int(param, *interval)
                else:
                    d[param] = trial.suggest_categorical(param, [*interval])
            model = self.model(**d)
            num_train_steps = int(len(self.train_data) * config.EPOCHS)
            warm_up_steps = int(num_train_steps*warmup_perc)

            learner = Learner(
                            config_name=self.test_name,
                            model=model,
                            metrics=self.metrics if self.metrics is not None else None,
                            lr=config.LR,
                            bs=config.BATCH_SIZE,
                            steps=num_train_steps,
                            warm_up_steps=warm_up_steps,
                            device=config.DEVICE,
                            verbose=self.verbose
                        )

            trainer = Trainer(
                        name=self.test_name, 
                        train_dataloader=self.train_data, 
                        valid_dataloader=self.valid_data, 
                        epochs=config.EPOCHS, 
                        measure=self.measure,
                        direction=self.direction,
                        configuration=learner, 
                        verbose=self.verbose,
                        return_predictions=create_plot
                    )

            results = trainer.execute()
            optim_param = results["best_metric"] 
            if create_plot:
                if optim_param >= 0.69:
                    labels = results["labels"]
                    preds = results["predictions"]
                    plot_roc_curve(labels, preds, savepath="graphs/roc_curve_sense_only")
            return optim_param
        return objective

    def find_params(self, n_trials, warmup_perc=0, create_plot=False, **params):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self._build_params(warmup_perc=warmup_perc, create_plot=create_plot, **params), n_trials=n_trials)
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        line = f"Best value: {best_value}; Best params: {best_params}; Best trial: {best_trial}\n\n"
        with open(os.path.join("results", self.test_name), "w") as f:
            f.write(line)
        if self.verbose:
            print(line)
        return best_params

if __name__ == '__main__':

    
    
    processor = WicProcessor()
    train_dataset = processor.build_dataset(examples_path="../data/WiC/train/train.data.txt", labels_path="../data/WiC/train/train.gold.txt")
    valid_dataset = processor.build_dataset(examples_path="../data/WiC/dev/dev.data.txt", labels_path="../data/WiC/dev/dev.gold.txt")


    train_data_loader = WiCDataLoader.build_batches(
        train_dataset, 
        batch_size=config.BATCH_SIZE
    )

    valid_data_loader = WiCDataLoader.build_batches(
        valid_dataset, 
        batch_size=config.BATCH_SIZE
    )
    metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}


    model = ParaphraseClassifierModelV2

    test = ParamOptimizer(model, "param_search_wic_no_sense", train_data_loader, valid_data_loader, measure="accuracy", direction="maximize", metrics=metrics)
    
    test.find_params(8, use_sense_embeddings=[False], train_model=[False], merge_strategy=["combine"], lr=[config.LR/10, config.LR*10], create_plot=True)
    
    
    """
    train_data_loader = utils.load_file("../dataset/cached/pawsx_en_train_dataloader_smart_16")
    valid_data_loader = utils.load_file("../dataset/cached/pawsx_en_valid_dataloader_smart_16")


    
    processor = dataset.ParallelPawsProcessor()
    langs = []
    tgt_train_paths = [f"../data/paws-x/{l}/translated_train.tsv" if l != "en" else f"../data/paws-x/{l}/train.tsv" for l in args.langs]
    tgt_valid_paths = [f"../data/paws-x/{l}/test_2k.tsv" for l in args.langs]
    train_dataset = processor.build_dataset(args.train_path, tgt_train_paths)
    valid_dataset = processor.build_dataset(args.valid_path, tgt_valid_paths)
    train_data_loader = dataset.SmartParaphraseDataloader.build_batches(train_dataset, args.batch_size, parallel_data=args.parallel_dataset)
   
    processor = dataset.PawsProcessor()
    
    train_dataset = processor.build_dataset("../data/paws/train.tsv", "../data/paws/train.tsv")
    valid_dataset = processor.build_dataset("../data/paws/test.tsv", "../data/paws/test.tsv")
    train_data_loader = dataset.SmartParaphraseDataloader.build_batches(train_dataset, 16)
    valid_data_loader = dataset.SmartParaphraseDataloader.build_batches(valid_dataset, 16)
    """

    
    train_data_loader = utils.load_file(f"../dataset/cached/pawsx_en_train_data_loader_smart_{config.BATCH_SIZE}")
    valid_data_loader = utils.load_file(f"../dataset/cached/pawsx_en_valid_data_loader_smart_{config.BATCH_SIZE}")
    
    sim_metrics = {"training": [SimilarityAveragePrecisionMeter, SimilarityAccuracyMeter], "validation": [SimilarityAveragePrecisionMeter, SimilarityAccuracyMeter]}
    model = SiameseSentenceEmbedder
    soft_metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}

    test = ParamOptimizer(
        model, 
        "training_siamese_paws_sense_online_contrastive_5_epochs", 
        train_data_loader, 
        valid_data_loader, 
        measure="ap", 
        direction="maximize", 
        metrics=sim_metrics)
    
    test.find_params(1, warmup_perc=0.1, use_sense_embeddings=[True], train_model=[True], loss=["online_contrastive"], merge_strategy=["substitute"])
    
    """
    train_data_loader = utils.load_file(f"../dataset/cached/pawsx_en_train_data_loader_smart_{config.BATCH_SIZE}_sentence_pairs")
    valid_data_loader = utils.load_file(f"../dataset/cached/pawsx_en_valid_data_loader_smart_{config.BATCH_SIZE}_sentence_pairs")
    
    model = MBERTClassifier
    soft_metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]}

    test = ParamOptimizer(model, "training_mbert_paws_sense_softmax_5_epochs", train_data_loader, valid_data_loader, measure="accuracy", direction="maximize", metrics=soft_metrics)
    
    test.find_params(1, warmup_perc=0.1, use_sense_embeddings=[True])
    """

    
    
    