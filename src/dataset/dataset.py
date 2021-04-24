from nltk import tag
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch.cuda import current_blas_handle
import src.configurations.config as config
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import src.utils.utils as utils
import torch
import json
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union, List, Dict, Tuple
import random
from tqdm import tqdm

class Dataset():
    def __init__(self, examples: List[Any]):
        self._examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

    def get_by_label(self, label: Any) -> List[Any]:
        return list(filter(lambda ex: ex.label == label, self.examples))

    def group_by_labels(self) -> Dict[Any, List[Any]]:
        dataset_labels = set(self.labels)
        groups = {}
        for lab in dataset_labels:
            groups[lab] = self.get_by_label(lab)
        return groups

    def split_dataset(self, test_perc: float=0.2) -> Tuple[List[Any], List[Any]]:
        """
        splits the dataset in train and test portions ensuring 
        all the labels are represented proportionally in each split
        """
        assert test_perc > 0 and test_perc < 1
        groups = self.group_by_labels()
        train_split = []
        test_split = []
        for label in groups:
            examples = groups[label]
            split_index = int(len(examples)*test_perc)
            test_examples = examples[0:split_index]
            train_examples = examples[split_index:]
            train_split.extend(train_examples)
            test_split.extend(test_examples)
        random.shuffle(train_split)
        random.shuffle(test_split)
        return train_split, test_split

    @property
    def examples(self):
        return self._examples

    @property
    def labels(self):
        return list(map(lambda ex: ex.label, self._examples))


class CrossValidationDataset:
    def __init__(self, train_splits, test_splits):
        self.__train_splits = train_splits
        self.__test_splits = test_splits

    def __getitem__(self, i):
        return self.train_splits[i], self.test_splits[i]

    def __len__(self):
        return len(self.train_splits)

    def train_test_split(self, split_n, test_size=0.2):
        train_split = self.__train_splits[split_n]
        test_split = self.__test_splits[split_n]
        dataset = Dataset(train_split.examples + test_split.examples)
        train_split, test_split, _, _ = train_test_split(dataset.examples, dataset.labels, test_size=test_size)

    
    @classmethod
    def create_folds(cls, dataset: Dataset, n_splits: int, fold_strategy: Optional[Any]=None, shuffle: bool=True, random_state: int=43):
        if fold_strategy is None:
            fold_strategy = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        train_splits = []
        test_splits = []
        for train_indexes, test_indexes in fold_strategy.split(dataset.examples, dataset.labels):
            train_examples = []
            test_examples = []
            for train_index, test_index in zip(train_indexes, test_indexes):
                train_examples.append(dataset.examples[train_index])
                test_examples.append(dataset.examples[test_index])
            train_splits.append(Dataset(train_examples))
            test_splits.append(Dataset(test_examples))
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


class DocumentCorpusExample():
    def __init__(self, document: str, sentences: List[str], id: int):
        self.document = document
        self.sentences = sentences
        self.id = id


class DocumentCorpusDataset:
    def __init__(self, documents: List[DocumentCorpusExample]):
        self.documents = documents

    def __getitem__(self, i):
        return self.documents[i]

    def __len__(self):
        return len(self.documents)

    @property
    def sentences(self):
        for doc in self.documents:
            for sent in doc.sentences:
                yield sent

    @property
    def articles(self):
        return list(map(lambda doc: ' '.join([doc.document] + doc.sentences), self.documents))

    @classmethod
    def from_tsv(cls, path):
        print("Parsing corpus. this may take a while")
        documents = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            iterator = tqdm(lines, total=len(lines))
            for i, line in enumerate(iterator):
                parts = line.split("\t")
                document = parts[0]
                sentences = parts[1:]
                documents.append(DocumentCorpusExample(document, sentences, i))
        print("Parsing complete!")
        return cls(documents)


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
        return Dataset(examples, labels)

    def build_document_dataset_from_paws(path):
        positive_examples = []
        negative_examples = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                parts = line.split("\t")
                id, sent1, sent2, label = parts
                label = int(label)
                sent_1_example = DocumentCorpusExample(sent1, i)
                sent_2_example = DocumentCorpusExample(sent2, i)
                if label == 1:
                    positive_examples += [sent_1_example, sent_2_example]
                else:
                    negative_examples += [sent_1_example, sent_2_example]
        return DocumentCorpusDataset(positive_examples, negative_examples)

@dataclass
class EmbeddingsFeatures:

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor = None

    @classmethod
    def from_dict(cls, dictionary, *args, **kwargs):
        return cls(
            dictionary["input_ids"],
            dictionary["attention_mask"],
            dictionary["token_type_ids"] if "token_type_ids" in dictionary else None,
            *args, 
            **kwargs
        )

    def to_dict(self):
        if self.token_type_ids is not None:
            return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "token_type_ids": self.token_type_ids
        }
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask
        }

    def generate_labels(self, model):
        with torch.no_grad():
            embeddings = model.encode(self)
        return embeddings

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.token_type_ids is not None:
            self.token_type_ids = self.token_type_ids.to(device)


@dataclass
class SenseEmbeddingsFeatures(EmbeddingsFeatures):
    tokens_indexes: List[torch.LongTensor] = None


@dataclass
class WordFeatures(EmbeddingsFeatures):
    indexes: List[torch.LongTensor] = None
    words: Union[List[str], None] = None


@dataclass
class DataFeatures:
    labels: torch.Tensor

    def to(self, device):
        self.labels = self.labels.to(device)


@dataclass
class DataLoaderFeatures(DataFeatures):
    embeddings_features: EmbeddingsFeatures

    def to(self, device):
        super().to(device)
        self.embeddings_features.to(device)

    def to_dict(self):
        return self.embeddings_features.to_dict()


@dataclass
class WordClassifierFeatures(DataFeatures):
    w1_features: WordFeatures
    w2_features: WordFeatures

    def to(self, device):
        super().to(device)
        self.w1_features.to(device)
        self.w2_features.to(device)
    

@dataclass
class SiameseDataLoaderFeatures(DataFeatures):
    sentence_1_features: EmbeddingsFeatures
    sentence_2_features: EmbeddingsFeatures
    tokens_indexes: Union[List[torch.LongTensor], None]
    src_sentences: Union[List[str], None] = None

    def to(self, device):
        super().to(device)
        self.sentence_1_features.to(device)
        self.sentence_2_features.to(device)


@dataclass
class ParallelDataLoaderFeatures:
    sentence_1_features: Union[EmbeddingsFeatures, Dict]
    sentence_2_features: Union[EmbeddingsFeatures, Dict]

    def to(self, device):
        if isinstance(self.sentence_1_features, dict):
            for k in self.sentence_1_features:
                self.sentence_1_features[k] = self.sentence_1_features[k].to(device)
            for k in self.sentence_2_features:
                self.sentence_2_features[k] = self.sentence_2_features[k].to(device)
        else:
            self.sentence_1_features.to(device)
            self.sentence_2_features.to(device)

    def generate_labels(self, model):
        embeddings = self.sentence_1_features.generate_labels(model)
        return embeddings


@dataclass
class DistillationDataLoaderFeatures:
    features: EmbeddingsFeatures
    sentences: Union[List[str], None] = None

    def to(self, device):
        self.features.to(device)


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
    def find_words_in_tokenized_sentence(tokenized_word, token_ids, current_pos=0):
        # Iterate through to find a matching sublist of the token_ids
        #example ['CLS', 'em', '##bed', '##ding', '##s', 'fl', '##ab', '##berg', '##ast', '##ed', 'SEP']
        for i in range(current_pos, len(token_ids)):
            try:
                if token_ids[i] == tokenized_word[0] and token_ids[i:i+len(tokenized_word)] == tokenized_word:
                    pos = (i,i+len(tokenized_word)-1)
                    return pos
            except IndexError:
                print(f"Word ids: {tokenized_word}")
                print(f"Sentence ids: {token_ids}")
                print(f"Current pos: {current_pos}")
                break
        

import random
class SmartParaphraseDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def build_batches(cls, dataset, batch_size, config, sentence_pairs=False, mode='standard', sbert_format=False):
        assert mode in ["sense_retrieval", "standard", "parallel", "tatoeba", "distillation", "word", "sequence"]
        if mode == "parallel":
            key = lambda x: max(len(x.get_sent1.strip().split(" ")), len(x.get_sent2.strip().split(" ")))
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_parallel(dataset, batch_size, config=config)
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
            batches = SmartParaphraseDataloader.smart_batching_standard(dataset, batch_size, config=config, sentence_pairs=sentence_pairs, sbert_format=sbert_format)
        if mode == "word":
            key = lambda x: len(x.sentence)
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_word(dataset, batch_size, config)
        if mode == "distillation":
            key = lambda x: len(x)
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_distillation(dataset, batch_size, config=config)
        if mode == "sequence":
            key = lambda x: len(x.content)
            dataset = sorted(dataset, key=key)
            batches = SmartParaphraseDataloader.smart_batching_sequence(dataset, batch_size, config=config)

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
    def build_indexes_mono(words, config, device):
        encoded_sent = config.tokenizer.encode(" ".join(words))
        encoded_words = []
        for w in words:
            encoded = config.tokenizer.encode(w)[1:-1]
            if encoded:
                encoded_words.append((w, encoded))
        positions = SmartParaphraseDataloader.find_tokens_positions(encoded_words, encoded_sent, device=device)
        return positions


    @staticmethod
    def find_tokens_positions(encoded_tokens, encoded_sentence, device, offset=None):
        visited = set()
        positions = [] #a list of tuples with token indexes in the tokenized sentence
        current_pos = 0
        for word, tokenized_word in encoded_tokens:  #encoded_tokens is a list of tuples (word, words_encoded_tokens)
            pos = DataLoader.find_words_in_tokenized_sentence(tokenized_word, encoded_sentence, current_pos=current_pos) 
            current_pos = pos[1]+1
            if pos:
                if offset is not None:
                    p_1 = pos[0] + offset
                    p_2 = pos[1] + offset
                    range_list = list(range(p_1, p_2+1))
                    el = torch.LongTensor(range_list).to(device)
                    positions.append(el)
                else:
                    range_list = list(range(pos[0], pos[1]+1))
                    el = torch.LongTensor(range_list).to(device)
                    positions.append(el)
        return positions

    @staticmethod
    def smart_batching_word(dataset, batch_size, config, sentence_pairs=False):
        batches = []
        tag_to_labels = dataset.tag_to_labels
        for i in range(0, len(dataset), batch_size):
            #take the examples of the current batches
            batch_examples = dataset[i:i+batch_size]
            #take the labels
            batch_labels = []
            sentences = []
            tokens_positions = []
            for ex in batch_examples:
                sentence = ex.sentence
                tokens = list(map(ex.tokens, lambda x: tag_to_labels[x]))
                sentences.append(sentence)
                labels = ex.tags
                batch_labels.append(torch.LongTensor(labels).to(config.device))
                positions = SmartParaphraseDataloader.build_indexes_mono(tokens, config, device=config.device)
                tokens_positions.append(positions)

            encoded = config.tokenizer(
                text=sentences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.sequence_max_len,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            pad_len = encoded["input_ids"].shape[1]
            

            features = WordFeatures(
                input_ids = encoded["input_ids"],
                attention_mask = encoded["attention_mask"],
                token_type_ids = encoded["token_type_ids"],
                indexes = tokens_positions,
                words = None
            )

            d = DataLoaderFeatures(
                embeddings_features=features,
                labels = torch.stack(batch_labels, dim=0).to(config.device)
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
    def smart_batching_standard(sorted_dataset, batch_size, config, sentence_pairs=False, sbert_format=False):
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

            batch_labels = torch.tensor(b_labels)

            if not sentence_pairs:
                encoded_dict_1 = config.tokenizer(
                    text=sentences_1,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=config.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                )

                encoded_dict_2 = config.tokenizer(
                    text=sentences_2,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=config.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                )

                sent_1_features = EmbeddingsFeatures.from_dict(encoded_dict_1)
                sent_2_features = EmbeddingsFeatures.from_dict(encoded_dict_2)

                if sbert_format:
                    d = SiameseDataLoaderFeatures(
                        sentence_1_features = encoded_dict_1,
                        sentence_2_features = encoded_dict_2 ,
                        labels = batch_labels,
                        tokens_indexes=None,
                    )
                    
                else:
                    d = SiameseDataLoaderFeatures(
                        sentence_1_features = sent_1_features,
                        sentence_2_features = sent_2_features ,
                        labels = batch_labels,
                        tokens_indexes=None,
                    )

            else:
                encoded_dict = config.tokenizer(
                text=sent_pairs,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.sequence_max_len,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
                )

                features = EmbeddingsFeatures.from_dict(encoded_dict)

                d = DataLoaderFeatures(
                        embeddings_features = features,
                        labels = batch_labels
                    )

            batches.append(d)
        return batches
    
    @staticmethod
    def smart_batching_parallel(sorted_dataset, batch_size, config):
        batches = []
        dataset = sorted_dataset
        while len(dataset) > 0:
            to_take = min(batch_size, len(dataset))
            select = random.randint(0, len(dataset)-to_take)
            batch = dataset[select:select+to_take]              
            
            src_sentences = []
            tgt_sentences = []
            for example in batch:
                src_sentences.append(example.get_sent1)
                tgt_sentences.append(example.get_sent2)

            encoded_dict_src = config.tokenizer(
                text=src_sentences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.sequence_max_len,
                return_attention_mask=True,
                #return_token_type_ids=True,
                return_tensors='pt'
            )
            
            encoded_dict_tgt = config.tokenizer_student(
                text=tgt_sentences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.sequence_max_len,
                return_attention_mask=True,
                #return_token_type_ids=True,
                return_tensors='pt'
            )
 
            features_src = EmbeddingsFeatures.from_dict(encoded_dict_src)
            features_tgt = EmbeddingsFeatures.from_dict(encoded_dict_tgt)

            d = ParallelDataLoaderFeatures(
                sentence_1_features = features_src,
                sentence_2_features = features_tgt
            )

            batches.append(d)

            del dataset[select:select+to_take]

        return batches

    @staticmethod
    def smart_batching_distillation(sorted_dataset, batch_size, config):
        batches = []
        dataset = sorted_dataset
        while len(dataset) > 0:
            to_take = min(batch_size, len(dataset))
            select = random.randint(0, len(dataset)-to_take)
            batch = dataset[select:select+to_take]             
            
            all_sentences = []
            for example in batch:
                all_sentences.append(example)

            random.shuffle(all_sentences) 
         
            encoded_dict = config.tokenizer(
                text=all_sentences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.sequence_max_len,
                return_attention_mask=True,
                #return_token_type_ids=True,
                return_tensors='pt'
            )

            features = EmbeddingsFeatures.from_dict(encoded_dict)

            batches.append(features)

            del dataset[select:select+to_take]

        return batches

    @staticmethod
    def smart_batching_sequence(sorted_dataset, batch_size, config):
        batches = []
        dataset = sorted_dataset
        while len(dataset) > 0:
            to_take = min(batch_size, len(dataset))
            select = random.randint(0, len(dataset)-to_take)
            batch = dataset[select:select+to_take]             
            labels = []
            documents = []
            for example in batch:
                documents.append(example.content)
                labels.append(example.label)
            encoded_dict = config.tokenizer(
                text=documents,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.sequence_max_len,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )

            features = EmbeddingsFeatures.from_dict(encoded_dict)
            dataloader_features = DataLoaderFeatures(
                embeddings_features = features,
                labels = torch.LongTensor(labels)
            )

            batches.append(dataloader_features)

            del dataset[select:select+to_take]

        return batches

