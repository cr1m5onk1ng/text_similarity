from .dataset import *

class QuoraExample(ParaphraseExample):
    def __init__(self, id, qid1, qid2, quest1, quest2, **kwargs):
        super().__init__(quest1, quest2, **kwargs)
        self.id = id
        self.qid1 = qid1
        self.qid2 = qid2

    @property
    def get_id(self):
        return self.id

    @property
    def get_qid1(self):
        return self.qid1

    @property
    def get_qid2(self):
        return self.qid2


class QuoraProcessor(ParaphraseProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, path):
        examples = []
        with open(path) as f:
            next(f)
            for line in f:
                #line = utils.remove_unnecessary_spaces(line)
                parts = line.split(',')
                id, qid1, qid2, question1, question2, is_duplicate = parts
                examples.append(QuoraExample(id, qid1, qid2, question1, question2))
        return examples

    def get_labels(self, path):
        labels = []
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.split(',')
                label = int(parts[-1])
                labels.append(bool(label))
        return labels