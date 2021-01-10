from .dataset import *

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


class GWSCDataset(ParaphraseDataset):
    def __init__(self, examples, labels):
        super().__init__(examples, labels)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i], self.labels[i]

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
        return cls(examples, labels)


class GWSCDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, batches):
        super().__init__(dataset, batch_size, batches)

    @classmethod
    def build_batches(cls, dataset, batch_size, evaluation=False):
        batches = []
        for i in range(0, len(dataset), batch_size):
            #take the examples of the current batches
            batch_examples = dataset.get_examples[i:i+batch_size]
            #take the labels
            labels = dataset.get_labels[i:i+batch_size]
            batch_labels = torch.FloatTensor(labels)
            lemmas_list_1 = []
            lemmas_list_2 = []
            sentences_1 = []
            sentences_2 = []
            w1_context1_positions = []
            w2_context1_positions = []
            w1_context2_positions = []
            w2_context2_positions = []
            for ex in batch_examples:
                sent1 = ex.get_sent1
                sent2 = ex.get_sent2
                sentences_1.append(sent1)
                sentences_2.append(sent2)
                word1 = ex.get_word1
                word2 = ex.get_word2
                lemmas_list_1.append(str(word1))
                lemmas_list_2.append(str(word2))
                word1_context1 = ex.get_word1_context1
                word2_context1 = ex.get_word2_context1
                word1_context2 = ex.get_word1_context2
                word2_context2 = ex.get_word2_context2
                w1_context1_idx = ex.get_word1_context1_idx
                w1_context2_idx = ex.get_word1_context2_idx
                w2_context1_idx = ex.get_word2_context1_idx
                w2_context2_idx = ex.get_word2_context2_idx
               
                sent1_tokenized = config.TOKENIZER(sent1)["input_ids"]
                sent2_tokenized = config.TOKENIZER(sent2)["input_ids"][1:]
                tokenized_w1_context1 = config.TOKENIZER(word1_context1)["input_ids"][1:-1]
                tokenized_w1_context2 = config.TOKENIZER(word1_context2)["input_ids"][1:-1]
                tokenized_w2_context1 = config.TOKENIZER(word2_context1)["input_ids"][1:-1]
                tokenized_w2_context2 = config.TOKENIZER(word2_context2)["input_ids"][1:-1]
 
                w1_context1_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w1_context1, sent1_tokenized)
                w2_context1_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w2_context1, sent1_tokenized)
                w1_context2_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w1_context2, sent2_tokenized)
                w2_context2_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w2_context2, sent2_tokenized)


                """
                w1_context2_idx_1_adjusted = w1_context2_token_positions[0] + len(sent1_tokenized)
                w1_context2_idx_2_adjusted = w1_context2_token_positions[1] + len(sent1_tokenized)
                w2_context2_idx_1_adjusted = w2_context2_token_positions[0] + len(sent1_tokenized)
                w2_context2_idx_2_adjusted = w2_context2_token_positions[1] + len(sent1_tokenized)
                """

                range_list_w1_context1 = list(range(w1_context1_token_positions[0], w1_context1_token_positions[1]+1))
                range_list_w1_context2 = list(range(w1_context2_token_positions[0], w1_context2_token_positions[1]+1))
                range_list_w2_context1 = list(range(w2_context1_token_positions[0], w2_context1_token_positions[1]+1))
                range_list_w2_context2 = list(range(w2_context2_token_positions[0], w2_context2_token_positions[1]+1))
                
                w1_context1_positions.append(torch.LongTensor(range_list_w1_context1))
                w1_context2_positions.append(torch.LongTensor(range_list_w1_context2))
                w2_context1_positions.append(torch.LongTensor(range_list_w2_context1))
                w2_context2_positions.append(torch.LongTensor(range_list_w2_context2))

            features_1 = config.TOKENIZER(
                text=sentences_1,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            features_2 = config.TOKENIZER(
                text=sentences_2,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=config.SEQUENCE_MAX_LENGTH,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            sentences_1_features = {
                "input_ids": features_1["input_ids"],
                "token_type_ids": features_1["token_type_ids"],
                "attention_mask": features_1["attention_mask"]
            }

            sentences_2_features = {
                "input_ids": features_2["input_ids"],
                "token_type_ids": features_2["token_type_ids"],
                "attention_mask": features_2["attention_mask"]
            }

            d = {
                "sentences_1_features": sentences_1_features,
                "sentences_2_features": sentences_2_features,
                "w1_context1_positions": w1_context1_positions,
                "w2_context1_positions": w2_context1_positions,
                "w1_context2_positions": w1_context2_positions,
                "w2_context2_positions": w2_context2_positions,
                "lemmas_list_1": lemmas_list_1,
                "lemmas_list_2": lemmas_list_2,
                "labels": batch_labels
            }

            batches.append(d)
        return cls(dataset, batch_size, batches)
