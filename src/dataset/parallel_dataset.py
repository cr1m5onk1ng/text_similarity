import gzip
from os import pathsep
from src.dataset.dataset import ParaphraseExample

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

    @property
    def get_tgt_sentences(self):
        return self.tgt_sentences

    @property
    def get_src_sentences(self):
        return self.src_sentences

    @staticmethod
    def _build_collate(path, max_examples):
        examples = []
        if max_examples is not None:
            cnt = 0
        with gzip.open(path, 'rt', encoding='utf8') if path.endswith('.gz') else open(path, 'r', encoding='utf8') as f:
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

    @classmethod
    def build_dataset(cls, filepaths, max_examples=None):
        examples = []
        for filepath in filepaths:
            examples.extend(ParallelDataset._build_collate(filepath, max_examples=max_examples))
        return cls(examples)