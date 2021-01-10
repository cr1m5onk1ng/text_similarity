from .dataset import *

class EntailmentExample(ParaphraseExample):
    def __init__(self, sent1, sent2, label, **kwargs):
        super().__init__(sent1, sent2, **kwargs)
        self.label = label

    @property
    def get_label(self):
        return self.label


class EntailmentDataset(ParaphraseDataset):
    def __init__(self, examples, labels):
        super().__init__(examples, labels)

    def __getitem__(self, i):
        return self.examples[i], self.labels[i]

    def __len__(self):
        return len(self.examples)

    @classmethod
    def build_dataset(cls, paths, snli_path=None):
        label2class = {"contradiction": 0, "entailment": 1, "neutral": 2}
        examples = []
        
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    try:
                        sent1, sent2, label = line.strip().split("\t")
                    except ValueError:
                        continue
                    label = label2class[label]
                    example = EntailmentExample(sent1, sent2, label)
                    examples.append(example)

        if snli_path is not None:
            with open(snli_path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    sent1 = str(obj["sentence1"]).strip()
                    sent2 = str(obj["sentence2"]).strip()   
                    label = str(obj["gold_label"]).strip()
                    if label not in label2class:
                        continue
                    label = label2class[label]
                    example = EntailmentExample(sent1, sent2, label)
                    examples.append(example)

        random.shuffle(examples)
        labels = [ex.get_label for ex in examples]
        return EntailmentDataset(examples, labels)