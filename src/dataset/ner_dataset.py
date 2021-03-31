from enum import unique
import torch

class NerExample():
    def __init__(self, sentence, tokens, tags):
        self.sentence = sentence
        self.tokens = tokens
        self.tags = tags

    def __repr__(self):
        return f"Sentence: {self.sentence} Tokens: {self.tokens} Tags: {self.tags} "

class NerDataset():
    def __init__(self, config, examples, tag_to_labels, return_tensors=True):
        self.config = config
        self.examples = examples
        self.tag_to_labels = tag_to_labels
        self.return_tensors = return_tensors

    def __getitem__(self, i):
        if self.return_tensors:
            tokens = self.examples[i].tokens
            tags = self.examples[i].tags

            ids = []
            target_tag =[]

            for i, s in enumerate(tokens):
                inputs = self.config.tokenizer.encode(
                    s,
                    add_special_tokens=False
                )
                input_len = len(inputs)
                ids.extend(inputs)
                target_tag.extend([self.tag_to_labels[tags[i]]] * input_len)

            ids = ids[:self.config.sequence_max_len - 2]
            target_tag = target_tag[:self.config.sequence_max_len - 2]

            ids = [101] + ids + [102]
            target_tag = [0] + target_tag + [0]

            mask = [1] * len(ids)
            token_type_ids = [0] * len(ids)

            padding_len = self.config.sequence_max_len - len(ids)

            ids = ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_tag = target_tag + ([0] * padding_len)

            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "labels": torch.tensor(target_tag, dtype=torch.long),
            }

        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def labels(self):
        for ex in self.examples:
            for tag in ex.tags:
                yield tag

    @property
    def all_tokens(self):
        for ex in self.examples:
            for tok in ex.tokens:
                yield tok


    @classmethod
    def from_conll(cls, path, config, skip_header=False, max_lines=None, return_tensors=False):
        n=0
        examples = []
        tag_to_labels = {}
        unique_tags = set()
        with open(path) as f:
            if skip_header:
                next(f)
            tokens = []
            tags = []
            for line in f.readlines():
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    tags.append(tag)
                    unique_tags.add(tag)
                n+=1
                if line == "\n":
                    sentence = " ".join(tokens)
                    examples.append(NerExample(sentence, tokens, tags)) 
                    tokens = []
                    tags = []
                if max_lines is not None and n > max_lines:
                    break
        for i, tag in enumerate(unique_tags):
            tag_to_labels[tag] = i
        print(f"Tags for the current dataset: {tag_to_labels}")
        return cls(examples=examples, tag_to_labels=tag_to_labels, return_tensors=return_tensors, config=config)

    def add_dataset(self, path, skip_header=False, max_lines=None):
        n=0
        examples = []
        with open(path) as f:
            if skip_header:
                next(f)
            tokens = []
            tags = []
            for line in f.readlines():
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    tags.append(tag)
                n+=1
                if line == "\n":
                    sentence = " ".join(tokens)
                    examples.append(NerExample(sentence, tokens, tags)) 
                    tokens = []
                    tags = []
                if max_lines is not None and n > max_lines:
                    break
        self.examples = self.examples + examples


if __name__ == "__main__":

    dataset = NerDataset.from_conll("../data/NER/entity-recognition-datasets-master/data/GUM/CONLL-format/data/train/gum-train.conll", skip_header=True, max_lines=None)

    print(dataset[:3])