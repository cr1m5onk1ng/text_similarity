from .dataset import *
import torch

class WicExample(ParaphraseExample):
    """single example from WiC dataset"""   
    def __init__(self, lemma, pos, idxs, sent1, sent2, **kwargs):
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

    @property
    def get_lemma(self):
        return self.lemma

    @property
    def get_pos(self):
        return self.pos

    @property
    def get_idxs(self):
        return self.idxs


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
                gold_entries.append(True)
            elif gold == 'F':
                gold_entries.append(False)
        return gold_entries

    def get_dataset_json(self, path):
        """ return a ParaphraseDataset with examples from json files """
        examples = []
        labels = []
        objects = []
        with open(path, "r") as f:
            for line in f:
                objects.append(json.loads(line))
        for obj in objects:
            lemma = str(obj["word"])
            sent1 = str(obj["sentence1"])
            sent2 = str(obj["sentence2"])
            label = bool(obj["label"])
            start1 = int(obj["start1"])
            start2 = int(obj["start2"])
            end1 = int(obj["end1"])
            end2 = int(obj["end2"])
            example = WicJsonExample(lemma, start1, start2, end1, end2, sent1, sent2)
            examples.append(example)
            labels.append(label)
        return ParaphraseDataset(examples, labels)


    @staticmethod    
    def get_dev_examples(path):
        """ returns a complete list of dev examples """
        raise NotImplementedError()

    @staticmethod    
    def get_test_examples(path):
        """ returns a complete list of test examples """
        raise NotImplementedError()


class WiCDataLoader(DataLoader):
    def __init__(self, batch_size, batches):
        super().__init__(batch_size, batches)

    @classmethod
    def build_batches(cls, dataset, batch_size, evaluation=False):
        batches = []
        for i in range(0, len(dataset), batch_size):
            #take the examples of the current batches
            batch_examples = dataset.get_examples[i:i+batch_size]
            #take the labels
            labels = dataset.get_labels[i:i+batch_size]
            batch_labels = torch.LongTensor(labels)
            lemmas_list = []
            sentence_pairs = []
            input_ids = []
            token_type_ids = []
            w1_tokens_positions = []
            w2_tokens_positions = []
            attention_mask = []
            for ex in batch_examples:
                lemma = ex.get_lemma
                lemmas_list.append(lemma)
                sent1 = ex.get_sent1
                sent2 = ex.get_sent2
                sentence_pairs.append([sent1, sent2])
                w1_idx = ex.get_idxs[0]
                w2_idx = ex.get_idxs[1]
                w1 = sent1.strip().split(" ")[w1_idx]
                w2 = sent2.strip().split(" ")[w2_idx]
                sent1_tokenized = config.CONFIG.tokenizer.encode(sent1)
                sent2_tokenized = config.CONFIG.tokenizer.encode(sent2)[1:]
                tokenized_w1 = config.CONFIG.tokenizer.encode(w1)[1:-1]
                tokenized_w2 = config.CONFIG.tokenizer.encode(w2)[1:-1]
                w1_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w1, sent1_tokenized)
                w2_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w2, sent2_tokenized)
                w2_idx1_adjusted = w2_token_positions[0] + len(sent1_tokenized)
                w2_idx2_adjusted = w2_token_positions[1] + len(sent1_tokenized)
                if w1_token_positions is None or w2_token_positions is None:
                    raise Exception("Something went wrong, words not found in tokenized sequence!")
                range_list_1 = list(range(w1_token_positions[0], w1_token_positions[1]+1))
                range_list_2 = list(range(w2_idx1_adjusted, w2_idx2_adjusted+1))
                w1_tokens_positions.append(torch.LongTensor(range_list_1).to(config.CONFIG.device))
                w2_tokens_positions.append(torch.LongTensor(range_list_2).to(config.CONFIG.device))

            encoded = config.CONFIG.tokenizer(
                text=sentence_pairs,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.CONFIG.sequence_max_len,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            embed_features_1 = WordFeatures.from_dict(encoded, indexes = w1_tokens_positions, words=lemmas_list)
            embed_features_2 = WordFeatures.from_dict(encoded, indexes = w2_tokens_positions, words=lemmas_list)

            d = WordClassifierFeatures(
                w1_features = embed_features_1,
                w2_features = embed_features_2,
                labels = batch_labels
            )

            d.to_device(config.CONFIG.device)

            batches.append(d)
        return cls(batch_size, batches)
    