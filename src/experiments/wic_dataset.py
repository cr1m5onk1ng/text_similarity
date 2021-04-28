from src.dataset.dataset import ParaphraseExample, Dataset

class ParaphraseProcessor():
    """ Base class for loading paraphrase data """
    def get_examples(self, path):
        raise NotImplementedError()

    def get_labels(self, path):
        raise NotImplementedError()

    def build_dataset(self, examples_path, labels_path):
        """ returns an instance of ParaphraseDatasat containing a list of examples"""
        examples = self.get_examples(examples_path)
        labels = self.get_labels(labels_path)
        for example, label in zip(examples, labels):
            example.label = label
        return Dataset(examples)


class WicExample(ParaphraseExample):
    """single example from WiC dataset"""   
    def __init__(self, lemma, pos, idxs, sent1, sent2, label,**kwargs):
        """
        Args:
            lemma: the lemma of the word in the two contexts
            pos: the part of speech of the word in the two contexts (Verb: V, Noun: N)
            idxs: indexes of the word in the first and second sentence
            sent1: firs sentence of the example
            sent2: second sentence of the example
        """
        super().__init__(sent1, sent2, **kwargs)
        self.lemma = lemma
        self.pos = pos
        self.idxs = idxs
        self.label = label

    @property
    def get_lemma(self):
        return self.lemma

    @property
    def get_pos(self):
        return self.pos

    @property
    def get_idxs(self):
        return self.idxs

    @property
    def get_label(self):
        return self.label


class WicProcessor(ParaphraseProcessor):
    def __init__(self):
        super().__init__()
    
    def get_examples(self, path):
        """returns a complete list of WiCExample objects"""
        data_entries = []
        for i, line in enumerate(open(path)):
            lemma, pos, idxs, sent1, sent2 = line.strip().split('\t')
            idx1, idx2 = list(map(int, idxs.split('-')))
            data_entries.append(WicExample(lemma, pos.lower(), (idx1, idx2), sent1, sent2))
        return data_entries

    def get_labels(self, path):
        """ returns the gold labels for the WiC examples """
        gold_entries = []
        for line in open(path):
            gold = line.strip()
            if gold == 'T':
                gold_entries.append(1)
            elif gold == 'F':
                gold_entries.append(0)
        return gold_entries


    @staticmethod    
    def get_dev_examples(path):
        """ returns a complete list of dev examples """
        raise NotImplementedError()

    @staticmethod    
    def get_test_examples(path):
        """ returns a complete list of test examples """
        raise NotImplementedError()