from src.configurations.config import load_file
from src.dataset.dataset import *

class EntailmentExample(ParaphraseExample):
    def __init__(self, sent1, sent2, label, **kwargs):
        super().__init__(sent1, sent2, **kwargs)
        self.label = label

    @property
    def get_label(self):
        return self.label


class EntailmentDataset(Dataset):
    def __init__(self, examples, labels):
        super().__init__(examples, labels)

    def __getitem__(self, i):
        return self.examples[i], self.labels[i]

    def __len__(self):
        return len(self.examples)

    @classmethod
    def build_dataset(cls, path, max_examples=None, all_nli=True, mode="train"):
        assert mode in ["test", "dev", "train"]
        label2class = {"contradiction": 0, "entailment": 1, "neutral": 2}
        examples = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()[1:]
            iterator = tqdm(lines, total=len(lines))
            if max_examples is not None:
                examples_read = 0
            print(f"Loading Dataset from {path}...")
            for line in iterator:
                if all_nli:
                    split, *_, sent1, sent2, label  = line.strip().split("\t")
                    if split != mode:
                        continue          
                else:
                    sent1, sent2, label  = line.strip().split("\t")
                try:
                    label = label2class[label]
                except KeyError:
                    print(f"Error for label: {label}")
                    continue
                example = EntailmentExample(sent1, sent2, label)
                examples.append(example)
                if max_examples is not None:
                    examples_read += 1
                    if examples_read >= max_examples:
                        break 
        random.shuffle(examples)
        labels = [ex.get_label for ex in examples]
        assert(len(labels)) == len(examples)
        print(f"Number of examples: {len(examples)}")
        return cls(examples, labels)

    def add_dataset(self, path):
        label2class = {"contradiction": 0, "entailment": 1, "neutral": 2}
        examples = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                label, sent1, sent2 = line.strip().split("\t")
                label = label2class[label]
                example = EntailmentExample(sent1, sent2, label)
                examples.append(example)
        self.examples.extend(examples)
        random.shuffle(self.examples)
                

