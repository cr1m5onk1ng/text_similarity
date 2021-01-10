from src.configurations import classifier_config as config
from src.configurations import embeddings_config as embeddings_config
from src.dataset.dataset import *
from src.utils.metrics import SimilarityCorrelationMeter
from src.models.modeling import GWSCModel
from src.evaluation.evaluators import ParaphraseEvaluator




if __name__ == "__main__":


    langs = ["en", "fi", "hr", "sl"]
    mono = ["sl"]
    print(f"Lang: {mono[0]}")
    examples_paths = [f"../data/GWSC/evaluation_kit_final/data/data_{l}.tsv" for l in mono]
    labels_paths = [f"../data/GWSC/evaluation_kit_final/res1/results_subtask1_{l}.tsv" for l in mono]

    dataset = GWSCDataset.build_dataset(examples_paths, labels_paths)

    data_loader = GWSCDataLoader.build_batches(dataset, config.BATCH_SIZE)

    model = GWSCModel(use_sense_embeddings=False, senses_as_features=False)

    

    metric = {"validation": [SimilarityCorrelationMeter]}

    evaluator = ParaphraseEvaluator(
        model = model,
        data_loader = data_loader,
        device = config.DEVICE,
        metrics = metric,
        fp16=True,
        verbose=True,
        return_predictions=True
    )

    results = evaluator.evaluate()

    




