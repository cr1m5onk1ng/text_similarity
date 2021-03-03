import gzip
from src.dataset.dataset import ParaphraseExample
import random

class ParallelExample(ParaphraseExample):
    def __init__(self, sent1, sent2, **kwargs):
        super().__init__(sent1, sent2, **kwargs)


class ParallelDataset():
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def _build_collate(path, max_examples, skip_header=False):
        examples = []
        if max_examples is not None:
            cnt = 0
        with gzip.open(path, 'rt', encoding='utf8') if path.endswith('.gz') else open(path, 'r', encoding='utf8') as f:
            if skip_header:
                next(f)
            for line in f:
                if max_examples is not None:
                    cnt+=1
                    if cnt > max_examples:
                        break
                splits = line.strip().split('\t')
                if len(splits) == 2:
                    src_sentence = splits[0]
                    tgt_sentence = splits[1]
                    examples.append(ParallelExample(src_sentence, tgt_sentence))
        return examples

    def add_dataset(self, path, max_examples=None, skip_header=False):
        self.examples.extend(ParallelDataset._build_collate(path, max_examples, skip_header))

    @classmethod
    def build_dataset(cls, filepaths, max_examples=None, skip_header=False):
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        examples = []
        for filepath in filepaths:
            examples.extend(ParallelDataset._build_collate(filepath, max_examples=max_examples, skip_header=skip_header))
        random.shuffle(examples)
        return cls(examples)