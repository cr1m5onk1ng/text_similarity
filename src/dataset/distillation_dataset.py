from typing import List
from src.dataset.dataset import Dataset
import random

class DistillationDataset:
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return len(self.sentences) 

    @classmethod
    def build_dataset(cls, parallel_datasets: List[Dataset], only_src=False, only_tgt=False, src_and_target=False):
        if not isinstance(parallel_datasets, list):
            parallel_datasets = [parallel_datasets]
        sentences = []
        for dataset in parallel_datasets:
            examples = dataset.examples
            for example in examples:
                if only_src:
                    sentences.append(example.get_sent1)
                if only_tgt:
                    sentences.append(example.get_sent2)
                if src_and_target:
                    sentences.append(example.get_sent1)
                    sentences.append(example.get_sent2)
                #sentences.append(example.get_sent2)
        random.shuffle(sentences)
        return cls(sentences)


