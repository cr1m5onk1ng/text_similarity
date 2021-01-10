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


class ParallelPawsDataset(ParaphraseDataset):
    def __init__(self, examples, labels, **kwargs):
        super().__init__(examples, labels, **kwargs)


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
        return ParaphraseDataset(examples, labels)     


class ParallelPawsProcessor(PawsProcessor):
    def __init__(self):
        super().__init__()

    def build_dataset(self, src_path, targets_paths, shuffle=False):
        labels = self.get_labels(src_path)
        src_examples = self.get_examples(src_path)
        examples = []
        for path in targets_paths:
            target_examples = self.get_examples(path)
            for src_example, tgt_example in zip(src_examples, target_examples):
                examples.append(ParallelPawsExample(src_example, tgt_example))
        return ParallelPawsDataset(examples, labels, shuffle=shuffle)


class PAWSDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, batches):
        super().__init__(dataset, batch_size, batches)
    
    @classmethod
    def build_batches(cls, dataset, batch_size):
        batches = []
        n_tokens = 0
        for i in range(0, len(dataset), batch_size):
            #take the examples of the current batches
            batch_examples = dataset.get_examples[i:i+batch_size]
            #take the labels
            labels = dataset.get_labels[i:i+batch_size]
            batch_labels = torch.LongTensor(labels)
            sentences_1 = []
            sentences_2 = []
            input_ids_1 = []
            token_type_ids_1 = []
            attention_mask_1 = []
            input_ids_2 = []
            token_type_ids_2 = []
            attention_mask_2 = []
            
            for ex in batch_examples:
                sent1 = ex.get_sent1
                #print("Sent1: {}".format(sent1))
                sent2 = ex.get_sent2
                #print("Sent2: {}".format(sent2))
                sentences_1.append(sent1)
                sentences_2.append(sent2)
                sent1_tokenized = config.TOKENIZER.encode(sent1)
                #print("Sent1 encoded: {}".format(sent1_tokenized))
                sent2_tokenized = config.TOKENIZER.encode(sent2)
                #print("Sent2 encoded: {}".format(sent2_tokenized))
                token_type_1 = ([1] * len(sent1_tokenized)) 
                token_type_2 = ([1] * len(sent2_tokenized)) 
                sent1_tokenized = sent1_tokenized
                sent2_tokenized = sent2_tokenized 
                #print("Current input ids: {}".format(current_input_ids))
                mask_1 = [1] * (len(sent1_tokenized)) 
                mask_2 = [1] * (len(sent2_tokenized)) 
                padding_1 = config.SEQUENCE_MAX_LENGTH - len(sent1_tokenized)
                padding_2 = config.SEQUENCE_MAX_LENGTH - len(sent2_tokenized)
                sent1_tokenized += [0]*padding_1
                sent2_tokenized += [0]*padding_2
                token_type_1 += [0]*padding_1
                token_type_2 += [0]*padding_2
                mask_1 += [0]*padding_1
                mask_2 += [0]*padding_2
                n_tokens += len(sent1_tokenized) + len(sent2_tokenized)
                input_ids_1.append(torch.LongTensor(sent1_tokenized))
                input_ids_2.append(torch.LongTensor(sent2_tokenized))
                token_type_ids_1.append(torch.LongTensor(token_type_1))
                token_type_ids_2.append(torch.LongTensor(token_type_2))
                attention_mask_1.append(torch.LongTensor(mask_1))
                attention_mask_2.append(torch.LongTensor(mask_2))
               
            input_ids_1 = torch.stack(input_ids_1, dim=0)
            token_type_ids_1 = torch.stack(token_type_ids_1, dim=0)
            attention_mask_1 = torch.stack(attention_mask_1, dim=0)
            input_ids_2 = torch.stack(input_ids_2, dim=0)
            token_type_ids_2 = torch.stack(token_type_ids_2, dim=0)
            attention_mask_2 = torch.stack(attention_mask_2, dim=0)
            
            sent1_features = {
                "input_ids": input_ids_1,
                "token_type_ids": token_type_ids_1,
                "attention_mask": attention_mask_1
            }

            sent2_features = {
                "input_ids": input_ids_2,
                "token_type_ids": token_type_ids_2,
                "attention_mask": attention_mask_2
            }

            d = {
                "sentence_1_features": sent1_features,
                "sentence_2_features": sent2_features,
                "sentences_1": sentences_1,
                "sentences_2": sentences_2,
                "labels": batch_labels
            }

            batches.append(d)
        print("Total number of tokens: {}".format(n_tokens))
        return cls(dataset, batch_size, batches)