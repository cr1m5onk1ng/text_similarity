from .gwsc_dataset import GWSCDataset
from src.dataset.dataset import SmartParaphraseDataloader
from src.models.word_encoder import GWSCModel
from src.utils.metrics import SimilarityCorrelationMeter
from src.evaluation.evaluators import ParaphraseEvaluator
import torch



if __name__ == "__main__":


    langs = ["en", "fi", "hr", "sl"]
    mono = ["sl"]
    print(f"Lang: {mono[0]}")
    examples_paths = [f"../data/GWSC/evaluation_kit_final/data/data_{l}.tsv" for l in mono]
    labels_paths = [f"../data/GWSC/evaluation_kit_final/res1/results_subtask1_{l}.tsv" for l in mono]

    dataset = GWSCDataset.build_dataset(examples_paths, labels_paths)

    data_loader = SmartParaphraseDataloader.build_batches(dataset, 16, mode="word")

    model = GWSCModel(use_sense_embeddings=False, senses_as_features=False)

    

    metric = {"validation": [SimilarityCorrelationMeter]}

    evaluator = ParaphraseEvaluator(
        model = model,
        data_loader = data_loader,
        device = torch.device("cuda"),
        metrics = metric,
        fp16=True,
        verbose=True,
        return_predictions=True
    )

    results = evaluator.evaluate()

    




