from .dataset import *

class PawsExample(ParaphraseExample):
    """ single example from PAWS dataset """
    def __init__(self, id, sent1, sent2, label, **kwargs):
        super().__init__(sent1, sent2, **kwargs)
        self.id = id
        self.label = label

    def __str__(self):
        return f"id: {self.id}; sent1: {self.sent1}; sent2: {self.sent2}"

    def __repr__(self):
        return f"id: {self.id}; sent1: {self.sent1}; sent2: {self.sent2}"

    @property
    def get_id(self):
        return self.id

    @property
    def get_label(self):
        return self.label


class ParallelPawsExample():
    def __init__(self, src_lang_example, tgt_lang_example):
        self.src_lang_example = src_lang_example
        self.tgt_lang_example = tgt_lang_example

    @property
    def get_src_example(self):
        return self.src_lang_example

    @property
    def get_tgt_example(self):
        return self.tgt_lang_example


class PawsProcessor(ParaphraseProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, paths):
        examples = []
        for path in paths:
            with open(path) as f:
                next(f)
                for line in f:
                    #line = utils.remove_unnecessary_spaces(line)
                    parts = line.split('\t')
                    id, sent1, sent2, label = parts
                    label = int(label)
                    examples.append(PawsExample(id, sent1, sent2, label))
        random.shuffle(examples)
        labels = [ex.get_label for ex in examples]
        return examples, labels

    def build_dataset(self, paths):
        examples, labels = self.get_examples(paths)   
        return Dataset(examples)     

