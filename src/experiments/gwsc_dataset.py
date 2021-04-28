from src.dataset.dataset import Dataset, ParaphraseExample
from src.utils import utils as utils

class GWSCExample(ParaphraseExample):
    def __init__(
        self, 
        word1, 
        word2, 
        context1, 
        context2, 
        word1_context1, 
        word2_context1, 
        word1_context2, 
        word2_context2,
        word1_context1_idx,
        word2_context1_idx,
        word1_context2_idx,
        word2_context2_idx, 
        **kwargs
        ):
       
        super().__init__(context1, context2, **kwargs)
        self.word1 = word1
        self.word2 = word2,
        self.word1_context1 = word1_context1
        self.word2_context1 = word2_context1
        self.word1_context2 = word1_context2
        self.word2_context2 = word2_context2
        self.word1_context1_idx = word1_context1_idx
        self.word2_context1_idx = word2_context1_idx
        self.word1_context2_idx = word1_context2_idx
        self.word2_context2_idx = word2_context2_idx

    @property
    def get_word1(self):
        return self.word1

    @property
    def get_word2(self):
        return self.word2

    @property
    def get_word1_context1(self):
        return self.word1_context1

    @property
    def get_word1_context2(self):
        return self.word1_context2
    
    @property
    def get_word2_context1(self):
        return self.word2_context1

    @property
    def get_word2_context2(self):
        return self.word2_context2

    @property
    def get_word1_context1_idx(self):
        return self.word1_context1_idx

    @property
    def get_word1_context2_idx(self):
        return self.word1_context2_idx
    
    @property
    def get_word2_context1_idx(self):
        return self.word2_context1_idx

    @property
    def get_word2_context2_idx(self):
        return self.word2_context2_idx 


class GWSCDataset(Dataset):
    def __init__(self, examples):
        super().__init__(examples)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

    @staticmethod
    def build_examples(paths):
        """returns a complete list of WiCExample objects"""
        data_entries = []
        for path in paths:
            with open(path) as f:
                next(f)
                for line in f:
                    word1, word2, context1, context2, word1_context1, word2_context1, word1_context2, word2_context2 = line.strip().split('\t')
                    word1 = str(word1)
                    word2 = str(word2)
                    context1 = utils.remove_html_tags(context1)
                    context1 = utils.pad_punctuation(context1)
                    context2 = utils.remove_html_tags(context2)
                    context2 = utils.pad_punctuation(context2)
                    context1_splitted = context1.strip().split(" ")
                    context2_splitted = context2.strip().split(" ")
                    word1_context1_idx = utils.find_in_list(context1_splitted, word1_context1)
                    word2_context1_idx = utils.find_in_list(context1_splitted, word2_context1)
                    word1_context2_idx = utils.find_in_list(context2_splitted, word1_context2)
                    word2_context2_idx = utils.find_in_list(context2_splitted, word2_context2)
                    data_entries.append(
                        GWSCExample(
                            word1, 
                            word2, 
                            context1, 
                            context2, 
                            word1_context1,
                            word2_context1, 
                            word1_context2, 
                            word2_context2,
                            word1_context1_idx,
                            word2_context1_idx,
                            word1_context2_idx,
                            word2_context2_idx
                            )
                        )
        return data_entries

    @staticmethod
    def build_labels(paths):
        """ returns the gold labels for the WiC examples """
        gold_entries = []
        for path in paths:
            with open(path) as f:
                next(f)
                for line in f:
                    gold = line.strip()
                    gold_entries.append(float(gold))
        return gold_entries

    @classmethod
    def build_dataset(cls, examples_paths, labels_paths):
        examples = GWSCDataset.build_examples(examples_paths)
        labels = GWSCDataset.build_labels(labels_paths)
        for example, label in zip(examples, labels):
            example.label = label
        return cls(examples)
