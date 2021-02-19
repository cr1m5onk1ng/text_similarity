import src.configurations.config as config
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
import src.utils.utils as utils
import torch
import json
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, List, Dict
import random


class KFoldStratifier:
    def __init__(self, train_splits, test_splits):
        self.__train_splits = train_splits
        self.__test_splits = test_splits

    def __getitem__(self, i):
        return self.train_splits[i], self.test_splits[i]

    def __len__(self):
        return len(self.train_splits)
    
    @classmethod
    def create_folds(cls, dataset, n_splits, stratifier=None, shuffle=True, random_state=1):
        if stratifier is None:
            stratifier = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        train_splits = []
        test_splits = []
        for train_indexes, test_indexes in stratifier.split(dataset.get_examples, dataset.get_labels):
            train_examples = []
            test_examples = []
            train_labels = []
            test_labels = []
            for train_index, test_index in zip(train_indexes, test_indexes):
                train_examples.append(dataset.get_examples[train_index])
                test_examples.append(dataset.get_examples[test_index])
                train_labels.append(dataset.get_labels[test_index])
                test_labels.append(dataset.get_labels[test_index])
            train_split = ParaphraseDataset(train_examples, train_labels)
            test_split = ParaphraseDataset(test_examples, test_labels)
            train_splits.append(train_split)
            test_splits.append(test_split)
        return cls(train_splits, test_splits)

    @property
    def train_splits(self):
        return self.__train_splits

    @property
    def test_splits(self):
        return self.__test_splits


class ParaphraseExample():
    """ Base class for all paraphrases examples """
    def __init__(self, sent1, sent2):
        self.sent1 = sent1
        self.sent2 = sent2

    @property
    def get_sent1(self):
        return self.sent1
    
    @property
    def get_sent2(self):
        return self.sent2


class ParallelExample():
    def __init__(self, src_lang_example, tgt_lang_example):
        self.src_lang_example = src_lang_example
        self.tgt_lang_example = tgt_lang_example

    @property
    def get_src_lang_example(self):
        return self.src_lang_example

    @property
    def get_tgt_lang_example(self):
        return self.tgt_lang_example


class Dataset():
    def __init__(self, examples, labels):
        self.examples, self.labels = examples, labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i], self.labels[i]

    @property
    def get_examples(self):
        return self.examples

    @property
    def get_labels(self):
        return self.labels


class ParaphraseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def build_mulilingual(cls, paths):
        examples = []
        for path in paths:
            with open(path) as f:
                next(f)
                for line in f:
                    parts = line.split('\t')
                    id, sent1, sent2, label = parts
                    label = int(label)
                    examples.append(PawsExample(id, sent1, sent2, label))
        random.shuffle(examples)
        labels = [ex.get_label for ex in examples]
            
        return cls(examples, labels)

import gzip
class ParallelDataset():
    def __init__(self, src_sentences, tgt_sentences):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences

    def __getitem__(self, i):
        return self.src_sentences[i], self.tgt_sentences[i]

    def __len__(self):
        return len(self.src_sentences)

    @property
    def get_tgt_sentences(self):
        return self.tgt_sentences

    @property
    def get_src_sentences(self):
        return self.src_sentences

    @classmethod
    def build_dataset(cls, filepaths):
        src_sentences = []
        trg_sentences = []
        for filepath in filepaths:
            with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, 'r', encoding='utf8') as f:
                for line in f:
                    splits = line.strip().split('\t')
                    if len(splits) == 2:
                        src_sentences.append(splits[0])
                        trg_sentences.append(splits[1])
        return cls(src_sentences, trg_sentences)


class ParallelParaphraseDataset():
    def __init__(self, **datasets):
        self.datasets = OrderedDict(datasets)
    
    def __len__(self):
        return len(self.datasets[self.datasets.keys()[0]])
    
    def __getitem__(self, i):
        return [(lang, data) for lang, data in self.datasets.items()]

    @property
    def get_item_by_lang(self, i, lang):
        return self.datasets[lang][i]


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
        return ParaphraseDataset(examples, labels)


class ParallelProcessor(ParaphraseProcessor):
    def __init__(self):
        super().__init__()
    

    def get_examples(self, src_path):
        labels = []

        src_examples = []
        with open(src_path) as f:
            next(f)
            for line in f:
                #line = utils.remove_unnecessary_spaces(line)
                parts = line.split('\t')
                id, sent1, sent2, label = parts
                src_examples.append(PawsExample(id, sent1, sent2))
                labels.append(label)
        
        return ParaphraseDataset(src_examples, labels)


    def build_dataset(self, paths, langs):
        datasets = {}
        for path, lang in zip(paths, langs):
            dataset = self.get_examples(path)
            datasets[lang] = dataset

        return ParallelParaphraseDataset(**datasets)


@dataclass
class EmbeddingsFeatures:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor

    @classmethod
    def from_dict(cls, dictionary, *args, **kwargs):
        return cls(
            dictionary["input_ids"],
            dictionary["token_type_ids"],
            dictionary["attention_mask"],
            *args, 
            **kwargs
        )

    def to_dict(self):
        return {
            "input_ids": self.input_ids,
            "token_type_ids": self.token_type_ids,
            "attention_mask": self.attention_mask
        }

    def to_device(self, device):
        self.input_ids = self.input_ids.to(device)
        self.token_type_ids = self.token_type_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)


@dataclass
class SenseEmbeddingsFeatures(EmbeddingsFeatures):
    tokens_indexes: List[torch.LongTensor]


@dataclass
class WordFeatures(EmbeddingsFeatures):
    indexes: List[torch.LongTensor]
    words: Union[List[str], None]


@dataclass
class DataFeatures:
    labels: torch.Tensor

    def to_device(self, device):
        self.labels = self.labels.to(device)


@dataclass
class DataLoaderFeatures(DataFeatures):
    embeddings_features: EmbeddingsFeatures

    def to_device(self, device):
        super().to_device(device)
        self.embeddings_features.to_device(device)
        

@dataclass
class WordClassifierFeatures(DataFeatures):
    w1_features: WordFeatures
    w2_features: WordFeatures

    def to_device(self, device):
        super().to_device(device)
        self.w1_features.to_device(device)
        self.w2_features.to_device(device)
    

@dataclass
class SiameseDataLoaderFeatures(DataFeatures):
    sentence_1_features: EmbeddingsFeatures
    sentence_2_features: EmbeddingsFeatures
    tokens_indexes: Union[List[torch.LongTensor], None]

    def to_device(self, device):
        super().to_device(device)
        self.sentence_1_features.to_device(device)
        self.sentence_2_features.to_device(device)


@dataclass
class ParallelDataLoaderFeatures(DataFeatures):
    src_sentence_features: SiameseDataLoaderFeatures
    tgt_sentence_features: SiameseDataLoaderFeatures

    def to_device(self, device):
        super().to_device(device)
        self.src_sentence_features.to_device(device)
        self.tgt_sentence_features.to_device(device)


class DataLoader:
    def __init__(self, batch_size, batches):
        self.batch_size = batch_size
        self.batches = batches
    
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]

    @property
    def get_batch_size(self):
        return self.batch_size

    @staticmethod 
    def find_word_in_tokenized_sentence(tokenized_word, token_ids, visited=None):
        """PROVARE CON UN METODO CHE SALVA GLI INDICI IN UN SET, E SE GLI INDICI SONO GIA STATI VISITATI LI SALTA """
        # Iterate through to find a matching sublist of the token_ids
        #example ['CLS', 'em', '##bed', '##ding', '##s', 'fl', '##ab', '##berg', '##ast', '##ed', 'SEP']
        for i in range(len(token_ids)):
            if token_ids[i] == tokenized_word[0] and token_ids[i:i+len(tokenized_word)] == tokenized_word:
                pos = (i,i+len(tokenized_word)-1)
                return pos

    # Si mantiene un set con le posizioni già visitate. Una volta raccolte le posizioni,
    # si ordinano e si mettono in una lista con la parola corrispondente associata
    # la lista contiene dunque triple (parola, posizione_1, posizione_2) ordinate per posizione
    @staticmethod 
    def find_words_in_tokenized_sentence(tokenized_word, token_ids, visited=None):
        # Iterate through to find a matching sublist of the token_ids
        #example ['CLS', 'em', '##bed', '##ding', '##s', 'fl', '##ab', '##berg', '##ast', '##ed', 'SEP']
        for i in range(len(token_ids)):
            if token_ids[i] == tokenized_word[0] and token_ids[i:i+len(tokenized_word)] == tokenized_word:
                pos = (i,i+len(tokenized_word)-1)
                if pos not in visited:
                    visited.add(pos)
                    return pos
        

import random
class SmartParaphraseDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def build_batches(cls, dataset, batch_size, sentence_pairs=False, mode='sense_retrieval'):
        assert mode in ["sense_retrieval", "standard", "parallel_data", "tatoeba"]
        if mode == "parallel_data":
            key = lambda x: len(x[0].get_src_example.get_sent1.strip().split(" ") + x[0].get_src_example.get_sent2.strip().split(" "))
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_sense(dataset, batch_size, parallel_data=True)
        if mode == "sense_retrieval":
            key = lambda x: max(len(x[0].get_sent1.strip().split(" ")), len(x[0].get_sent2.strip().split(" ")))
            dataset = sorted(dataset, key=key)
            if sentence_pairs:
                batches = SmartParaphraseDataloader.smart_batching_sense_sentence_pairs(dataset, batch_size)
            else:
                batches = SmartParaphraseDataloader.smart_batching_sense(dataset, batch_size)
        if mode == "standard":
            key = lambda x: max(len(x[0].get_sent1.strip().split(" ")), len(x[0].get_sent2.strip().split(" ")))
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_standard(dataset, batch_size, sentence_pairs=sentence_pairs)
        if mode == "tatoeba":
            key = lambda x: len(x[0].strip().split(" ") + x[1].strip().split(" "))
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_parallel(dataset, batch_size)

        return cls(batch_size, batches)

    @staticmethod
    def build_indexes_combine(sent1, sent2):
        encoded_sent_1 = config.TOKENIZER.encode(sent1)
        encoded_sent_2 = config.TOKENIZER.encode(sent2)[1:]
        encoded_pair = encoded_sent_1 + encoded_sent_2
        all_words = sent1.split(" ") + sent2.split(" ")
        encoded_words = []
        for w in all_words:
            encoded = config.TOKENIZER.encode(w)[1:-1]
            encoded_words.append((w, encoded))
        return SmartParaphraseDataloader.find_tokens_positions(encoded_words, encoded_pair)

    @staticmethod
    def build_indexes(sent1, sent2):
        encoded_sent_1 = config.TOKENIZER.encode(sent1)
        encoded_sent_2 = config.TOKENIZER.encode(sent2)[1:]
        words_1 = sent1.split(" ")
        words_2 = sent2.split(" ")
        encoded_words_1 = []
        encoded_words_2 = []
        for w1, w2 in zip(words_1, words_2):
            encoded_1 = config.TOKENIZER.encode(w1)[1:-1]
            encoded_2 = config.TOKENIZER.encode(w2)[1:-1]
            encoded_words_1.append((w1, encoded_1))
            encoded_words_2.append((w2, encoded_2))
        positions_1 = SmartParaphraseDataloader.find_tokens_positions(encoded_words_1, encoded_sent_1)
        positions_2 = SmartParaphraseDataloader.find_tokens_positions(encoded_words_2, encoded_sent_2)
        return positions_1, positions_2

    @staticmethod
    def find_tokens_positions(encoded_tokens, encoded_sentence, offset=None):
        visited = set()
        positions = [] #a list of tuples with token indexes in the tokenized sentence
        for word, tokenized_word in encoded_tokens:  #encoded_tokens is a list of tuples (word, words_encoded_tokens)
            pos = DataLoader.find_words_in_tokenized_sentence(tokenized_word, encoded_sentence, visited) 
            if pos:
                if offset is not None:
                    p_1 = pos[0] + offset
                    p_2 = pos[1] + offset
                    range_list = list(range(p_1, p_2+1))
                    el = (word, torch.LongTensor(range_list))
                    positions.append(el)
                else:
                    range_list = list(range(pos[0], pos[1]+1))
                    el = (word, torch.LongTensor(range_list))
                    positions.append(el)
        return positions

    @staticmethod
    def smart_batching_sense_sentence_pairs(sorted_dataset, batch_size, parallel_data=False):
        batches = []
        dataset = sorted_dataset
        while len(dataset) > 0:
            to_take = min(batch_size, len(dataset))
            select = random.randint(0, len(dataset)-to_take)
            batch = dataset[select:select+to_take]           
            
            b_labels = []
         
            sent_pairs = []
            sentences_words_positions = []
            for ex, label in batch:
                if parallel_data:
                    sent1 = ex.get_tgt_example.get_sent1.strip()
                    sent2 = ex.get_tgt_example.get_sent2.strip()
                    sentences_1.append(sent1)
                    sentences_2.append(sent2)
                    sent_1_src = ex.get_src_example.get_sent1
                    sent_2_src = ex.get_src_example.get_sent2
                    src_sentences_1.append(sent_1_src)
                    src_sentences_2.append(sent_2_src)
                else:
                    sent1 = ex.get_sent1.strip()
                    sent2 = ex.get_sent2.strip()
                    words_positions = SmartParaphraseDataloader.build_indexes_combine(sent1, sent2)
                    
                    sentences_words_positions.append(words_positions)
                 
                    sent_pairs.append([sent1, sent2])
                   
                b_labels.append(label)

            del dataset[select:select+to_take]

            encoded_dict = config.TOKENIZER(
                text=sent_pairs,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            batch_labels = torch.LongTensor(b_labels)

            features = EmbeddingsFeatures.from_dict(encoded_dict)

            d = SiameseDataLoaderFeatures(
                embeddings_feature = features,
                tokens_indexes = sentences_words_positions,
                labels = batch_labels
            )

            batches.append(d)

        return batches


    @staticmethod
    def smart_batching_sense(sorted_dataset, batch_size, sentence_pairs=False):
        batches = []
        dataset = sorted_dataset
        while len(dataset) > 0:
            to_take = min(batch_size, len(dataset))
            select = random.randint(0, len(dataset)-to_take)
            batch = dataset[select:select+to_take]           
            
            b_labels = []
            src_sentences_1 = []
            src_sentences_2 = []
            sentences_1 = []
            sentences_2 = []
            sent_pairs = []
            sentences_1_words_positions = []
            sentences_2_words_positions = []
            for ex, label in batch:
                sent1 = ex.get_sent1.strip()
                sent2 = ex.get_sent2.strip()
                
                positions_1, positions_2 = SmartParaphraseDataloader.build_indexes(sent1, sent2)
                
                sentences_1_words_positions.append(positions_1)
                sentences_2_words_positions.append(positions_2)
                if sentence_pairs:
                    sent_pairs.append([sent1, sent2])
                sentences_1.append(sent1)
                sentences_2.append(sent2)
                b_labels.append(label)

            del dataset[select:select+to_take]

            batch_labels = torch.LongTensor(b_labels)

            encoded_dict_1 = config.TOKENIZER(
                text=sentences_1,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            encoded_dict_2 = config.TOKENIZER(
                text=sentences_2,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            sent_1_features = SenseEmbeddingsFeatures.from_dict(encoded_dict_1, sentences_1_words_positions)
            sent_2_features = SenseEmbeddingsFeatures.from_dict(encoded_dict_2, sentences_2_words_positions)

            d = SiameseDataLoaderFeatures(
                sentence_1_features = sent_1_features,
                sentence_2_features = sent_2_features,
                labels = batch_labels
            )

            batches.append(d)

        return batches
    
    @staticmethod
    def smart_batching_standard(sorted_dataset, batch_size, sentence_pairs=False):
        batches = []
        dataset = sorted_dataset
        while len(dataset) > 0:
            to_take = min(batch_size, len(dataset))
            select = random.randint(0, len(dataset)-to_take)
            batch = dataset[select:select+to_take]           
            
            b_labels = []
            sentences_1 = []
            sentences_2 = []
            sent_pairs = []
            for ex, label in batch:
                if not sentence_pairs:
                    sent1 = ex.get_sent1
                    sent2 = ex.get_sent2
                    sentences_1.append(sent1)
                    sentences_2.append(sent2)
                else:
                    sent1 = ex.get_sent1
                    sent2 = ex.get_sent2
                    sent_pairs.append([sent1, sent2])

                b_labels.append(label)

            del dataset[select:select+to_take]

            batch_labels = torch.LongTensor(b_labels)

            if not sentence_pairs:
                encoded_dict_1 = config.CONFIG.tokenizer(
                    text=sentences_1,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=config.CONFIG.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                )

                encoded_dict_2 = config.CONFIG.tokenizer(
                    text=sentences_2,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=config.CONFIG.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                )

                sent_1_features = EmbeddingsFeatures.from_dict(encoded_dict_1)
                sent_2_features = EmbeddingsFeatures.from_dict(encoded_dict_2)

                d = SiameseDataLoaderFeatures(
                    sentence_1_features = sent_1_features,
                    sentence_2_features = sent_2_features,
                    labels = batch_labels,
                    tokens_indexes=None
                )

            else:
                encoded_dict = config.CONFIG.tokenizer(
                text=sent_pairs,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.CONFIG.sequence_max_len,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
                )

                features = EmbeddingsFeatures.from_dict(encoded_dict)

                d = DataLoaderFeatures(
                        embeddings_feature = features,
                        labels = batch_labels
                    )

            batches.append(d)
        return batches
    
    @staticmethod
    def smart_batching_parallel(sorted_dataset, batch_size):
        batches = []
        dataset = sorted_dataset
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]           
            
            src_sentences = []
            tgt_sentences = []
            sent_pairs = []
            for src_sentence, tgt_sentence in batch:
                src_sentences.append(src_sentence)
                tgt_sentences.append(tgt_sentence)
            
            encoded_dict_1 = config.TOKENIZER(
                text=src_sentences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            encoded_dict_2 = config.TOKENIZER(
                text=tgt_sentences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            sent1_features = {
                "input_ids": encoded_dict_1["input_ids"].to(config.DEVICE),
                "token_type_ids": encoded_dict_1["token_type_ids"].to(config.DEVICE),
                "attention_mask": encoded_dict_1["attention_mask"].to(config.DEVICE)
            }

            sent2_features = {
                "input_ids": encoded_dict_2["input_ids"].to(config.DEVICE),
                "token_type_ids": encoded_dict_2["token_type_ids"].to(config.DEVICE),
                "attention_mask": encoded_dict_2["attention_mask"].to(config.DEVICE)
            }
    
            d = {
                "sentence_1_features": sent1_features,
                "sentence_2_features": sent2_features,
                "sentences_1": src_sentences,
                "sentences_2": tgt_sentences
            }

            batches.append(d)

        return batches






if __name__ == "__main__":
    """
    processor = PawsProcessor()
    train_paths = ["../data/paws/train.tsv", "../data/paws/train.tsv"]
    valid_pathts = ["../data/paws/test.tsv", "../data/paws/test.tsv"]
    train_dataset = processor.build_dataset([train_paths[0]])
    valid_dataset = processor.build_dataset([valid_pathts[0]])

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, sentence_pairs=False)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, sentence_pairs=False)
    utils.save_file(train_data_loader, "cached/paws", "train_en_sense_16") 
    utils.save_file(valid_data_loader, "cached/paws", "valid_en_sense_16") 
    """
    """
    valid_data_loader = utils.load_file("cached/paws/valid_en_sense_16") 
    print(valid_data_loader[0]["sentences_1_words_positions"])
    print()
    print(valid_data_loader[0]["sentences_2_words_positions"])
    print()
    print(len(valid_data_loader[0]["sentences_1_words_positions"]))
    print()
    print(len(valid_data_loader[0]["sentences_2_words_positions"]))

    """
    """
    processor = PawsProcessor()
    train_paths = ["../data/paws/train.tsv", "../data/paws/train.tsv"]
    valid_pathts = ["../data/paws/test.tsv", "../data/paws/test.tsv"]
    train_dataset = processor.build_dataset([train_paths[0]])
    valid_dataset = processor.build_dataset([valid_pathts[0]])

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, sentence_pairs=True)
    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, sentence_pairs=True)
    utils.save_file(train_data_loader, "cached/paws", "train_pairs_en_16") 
    utils.save_file(valid_data_loader, "cached/paws", "valid_pairs_en_16") 
    """
    
    """

    langs = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    paths = [f"../data/xnli/dev-{l}.tsv" for l in langs]
    train_path = ['../data/xnli/train-en.tsv']
    paths += train_path
    
    train_dataset = EntailmentDataset.build_dataset(paths=paths, snli_path="../data/snli/snli_1.0_train.jsonl")
    valid_dataset = EntailmentDataset.build_dataset(["../data/xnli/dev-en.tsv"])

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 8, mode="standard")

    valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 8, mode="standard")

    utils.save_file(train_data_loader, "cached/nli", "nli_train_all_languages_8_xlm") #combination of snli and mnli
    utils.save_file(valid_data_loader, "cached/nli", "nli_valid_8_xlm") #english dev set from nli
    print(valid_data_loader[0])
    """



    """
    train_data_loader = utils.load_file("cached/nli/nli_train_16")

    print(train_data_loader[0])
    """

    
    #processor = PawsProcessor()
    #root = "../data/paws-x"
    #for lang in os.listdir(root):
    #    lang_dir = os.path.join(root, lang)
    #    print(lang_dir)
    #    for f in os.listdir(lang_dir):
    #        file = os.path.join(lang_dir, f)
    #        if os.path.isfile(file):
    #            print(f)
    #            p = os.path.join(lang_dir, f)
    #            dataset = processor.build_dataset(examples_path=p, labels_path=p)
    #            #dataloader = PAWSDataLoader(dataset, batch_size = 16)
    #            utils.save_file(dataset, "../data/cached", lang+"_"+f[:-4])
    #

    """
    processor = PawsProcessor()
    
    langs = ['de', 'en', 'es', 'fr', 'ja', 'ko', 'zh']
    paths = [f'../data/paws-x/{l}/test_2k.tsv' for l in langs]
    train_path = ['../data/paws-x/en/train.tsv']
    paths = paths + train_path
    train_dataset = ParaphraseDataset.build_mulilingual(paths)

    train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="standard", sentence_pairs=True)
   
    utils.save_file(train_data_loader, "../dataset/cached", "pawsx_train_data_all_languages_sent_pairs_16")

    print(train_data_loader[0])
    """

    #
    #
    #language_pairs = ['eng-ara', 'eng-deu', 'eng-fra', 'eng-ita', 'eng-spa', 'eng-tur']
    #train_paths = [f'../data/tatoeba/parallel-sentences/Tatoeba-{pair}-train.tsv.gz' for pair in language_pairs ]
    #dev_paths = [f'../data/tatoeba/parallel-sentences/Tatoeba-{pair}-dev.tsv.gz' for pair in language_pairs]

    ##train_dataset = ParallelDataset.build_dataset(train_paths)
    #valid_dataset = ParallelDataset.build_dataset(dev_paths)

    ##train_data_loader = SmartParaphraseDataloader.build_batches(train_dataset, 16, mode="tatoeba")
    #valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 32, mode="tatoeba")

    ##utils.save_file(train_data_loader, 'cached/tatoeba', 'tatoeba_train_smart_32')
    #utils.save_file(valid_data_loader, 'cached/tatoeba', 'tatoeba_valid_smart_32')
    #
    #language_pairs = ['eng-ara', 'eng-deu', 'eng-fra', 'eng-ita', 'eng-spa', 'eng-tur']
    #train_paths = [f'../data/tatoeba/parallel-sentences/Tatoeba-{pair}-train.tsv.gz' for pair in language_pairs ]
    #dev_paths = [f'../data/tatoeba/parallel-sentences/Tatoeba-{pair}-dev.tsv.gz' for pair in language_pairs]

    ##train_dataset = ParallelDataset.build_dataset(train_paths)
    #valid_dataset = ParallelDataset.build_dataset(dev_paths)
    #print(len(valid_dataset))
    #
    #tatoeba = utils.load_file("cached/tatoeba/tatoeba_valid_smart_16")
    #print(tatoeba[0])
    #

    """
    langs = ['de', 'en', 'es', 'fr', 'ja', 'ko', 'zh']
    paths = [f'../data/paws-x/{l}/test_2k.tsv' for l in langs]
    valid_dataset = ParaphraseDataset.build_mulilingual(paths)
    valid_dataloader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="standard", sentence_pairs=True)
    utils.save_file(valid_dataloader, 'cached', 'pawsx_test_all_languages_16')
    print(valid_dataloader[0])
    """
    #
    #
    #processor = WicProcessor()
    ##train_dataset = processor.build_dataset(examples_path="../data/WiC/train/train.data.txt", labels_path="../data/WiC/train/train.gold.txt")
    #valid_dataset = processor.build_dataset(examples_path="../data/WiC/dev/dev.data.txt", labels_path="../data/WiC/dev/dev.gold.txt")

    #valid_data_loader = WiCDataLoader.build_batches(
    #    valid_dataset, 
    #    batch_size=config.BATCH_SIZE
    #)
    #
    #print(f"Input IDS: {valid_data_loader[0]['input_ids']}\n\n")
    #print(f"w1 indexes: {valid_data_loader[0]['w1_idxs']}\n\n")
    #print(f"w2 indexes: {valid_data_loader[0]['w2_idxs']}")
    #
   