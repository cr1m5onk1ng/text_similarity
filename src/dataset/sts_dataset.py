from src.dataset.dataset import ParaphraseDataset, ParaphraseExample
from tqdm import tqdm
import random
import csv

class StsExample(ParaphraseExample):
    def __init__(self, sent1, sent2, label, *args, **kwargs):
        super().__init__(sent1, sent2, *args, **kwargs)
        self.label = label

    @property
    def get_label(self):
        return self.label

class StsDataset(ParaphraseDataset):
    def __init__(self, examples, labels, *args, **kwargs):
        super().__init__(examples, labels, *args, **kwargs)

    def __getitem__(self, i):
        return self.examples[i], self.labels[i]

    def __len__(self):
        return len(self.examples)

    @classmethod
    def build_dataset(cls, path, mode="train"):
        assert mode in ["train", "test", "dev"]
        examples = []
        with open(path, 'r', encoding='utf8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                sent1 = row['sentence1']
                sent2 = row['sentence2']
                if mode == "train" and row['split'] != "train":
                    continue
                if mode == "test" and row['split'] != "test":
                    continue
                if mode == "dev" and row['split'] != "dev":
                    continue
                example = StsExample(sent1, sent2, score)
                examples.append(example)
        random.shuffle(examples)
        labels = [ex.get_label for ex in examples]
        assert(len(labels)) == len(examples)
        print(f"Number of examples: {len(examples)}")
        return cls(examples, labels)