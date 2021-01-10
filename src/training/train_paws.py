import argparse
import torch
from torch import nn 
from torch.nn import functional as F
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
from tqdm import tqdm
from torch.cuda import amp
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup



##### --------- CONSTANTS --------- #####

def load_file(path):
    file_name = open(path, 'rb')
    d = pickle.load(file_name)
    file_name.close()
    return d

EMBEDDINGS_PATHS = {
    "ares_multi": "../embeddings/ares_embed_map",
    "ares_mono": "../embeddings/ares_embed_map_large",
    "lmms": "../embeddings/lmms_embed_map"
}

MODELS = {
    "base": "bert-base-cased",
    "large": "bert-large-cased",
    "multi": "bert-base-multilingual-cased"
}

LEARNING_RATES = {
    "pawsx": 2e-5,
    "wic": 1e-4,
}

MODEL = MODELS["multi"]
MODEL_PATH = "../models"
SEQUENCE_MAX_LENGTH = 128
DROPOUT_PROB = 0.1
LR = LEARNING_RATES["pawsx"]
NUM_LAYERS = 1
BATCH_SIZE = 16
EPOCHS = 1
OPTIMIZER = 'adam'
DEVICE = torch.device("cuda")
EMBEDDING_MAP = load_file(EMBEDDINGS_PATHS["ares_multi"])
BNIDS_MAP = load_file("../data/bnids_map")
TOKENIZER = transformers.BertTokenizer.from_pretrained(MODEL)


DIMENSIONS_MAP = {
    "ares_multi": 768*2,
    "ares_mono": 1024*2,
    "lmms": 1024
}

PRETRAINED_EMBED_PATH = '../embeddings/sensembert+lmms.svd512.synset-centroid.vec'
EMBEDDINGS_MAP_PATH = '../embeddings'
SENSE_EMBEDDING_DIMENSION = DIMENSIONS_MAP["ares_multi"]





####### ---------------- UTILS ------------------ #######

def save_file(file, path, name):
    file_name = open(os.path.join(path, name), 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def l2norm(tensor):
    norm = torch.norm(tensor, p=2)
    tensor /= norm

def scale_norm(weight):
    data = tensor.weight.data
    mean = data.mean(0).view(1, data.size(1))
    data /= mean
    tensor.weight.data = data
    return tensor

def mean_norm(weight):
    data = tensor.weight.data
    mean = data.mean(0).view(1, data.size(1))
    std = data.std(0).view(1, data.size(1))
    data -= mean
    data /= std
    tensor.weight.data = data
    return tensor

def combine_tensors(tensors, strategy='avg'):
    assert strategy in ["avg", "sum"]
    assert isinstance(tensors, list)
    if not tensors:
        return None
    stacked = torch.stack(tensors, dim=0)
    if strategy == 'avg':
        return torch.mean(stacked, dim=0)
    elif strategy == 'sum':
        return torch.sum(stacked, dim=0)

def remove_unnecessary_spaces(out_string):
    assert isinstance(out_string, str)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace("n't", " not")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string

def most_similar_vectors(vector, vectors, n=1):
    """ given a vector, finds the n most similar vectors using a similarity metric"""
    #expanding the same vector for efficiency
    expanded = vector.expand_as(vectors)
    scores = F.cosine_similarity(expanded, vectors)
    scores = [(i, score) for i, score in enumerate(scores)]
    ordered = list(sorted(scores, key=lambda x: x[1], reverse=True))
    res = []
    for i in range(n):
        res.append(vectors[ordered[i][0]])
    return res

def most_similar_senses(sense_vector_map, contextualized_vector):
    scores = []
    if len(contextualized_vector.shape) < 2:
        contextualized_vector = contextualized_vector.unsqueeze(0)
    scores = []
    for sense, v in sense_vector_map.items():
        if len(v.shape) < 2:
            v = v.unsqueeze(0)
        score = F.cosine_similarity(vector, v)
        scores.append((v, sense, score))
    ordered = list(sorted(scores, key=lambda x: x[2], reverse=True))
    res = []
    for i in range(n):
        res.append(ordered[i])
    return res

def word_to_wn_offsets(word, include_hypernyms=False, return_mapping=False, use_sense_keys=True, map_to_bnids=True):
    """ given a word, returns the offsets of the synsets related to the word """
    offsets = []
    offset_to_synset = {}
    if map_to_bnids:
        bnids_map = BNIDS_MAP
    for synset in wn.synsets(word):
        if not use_sense_keys:
            offset = "wn:" + str(synset.offset()) + str(synset.pos())
            if return_mapping:
                offset_to_synset[offset] = synset
            else:
                offsets.append(offset)
        else:
            for lemmas in synset.lemmas():
                key = lemmas.key()
                if not return_mapping:
                    if map_to_bnids:
                        if key in bnids_map:
                            offsets.append(bnids_map[key])
                    else:
                        offsets.append(key)
                else:
                    offset_to_synset[synset] = key
        if include_hypernyms:
            offsets.extend(get_synset_hypernyms_offsets(synset))
    return offsets if not return_mapping else offset_to_synset

def get_synset_hypernyms_offsets(synset):
    """ given a synset, returns the offsets of its related hypernyms """
    hypernyms = []
    for hypernym in synset.hypernyms():
        hypernyms.append("wn:" + str(hypernym.offset()) + str(hypernym.pos()))
    return hypernyms

def word_to_sense_embeddings(word, embed_map, include_hypernyms=False, return_senses=False, map_to_bnids=True):
    """takes a word and returns its sense embeddings representations """
    offsets = word_to_wn_offsets(word, include_hypernyms=include_hypernyms, return_mapping=return_senses, map_to_bnids=map_to_bnids)
    vectors = []
    sense_to_vector = {}
    for offset in offsets:
        if offset in embed_map:
            #print("sense found!")
            vectors.append(embed_map[offset].to(DEVICE))
            if return_senses:
                sense_to_vector[offsets[offset]] = embed_map[offset].to(DEVICE)
    if return_senses:
        return sense_to_vector
    return vectors

def words_to_sense_embeddings(words, vectors, embed_map):
    return [word_to_sense_embedding(w, vector, EMBEDDING_MAP) for w, vector in zip(words, vectors)]

def word_to_sense_embedding(word, vector, embed_map, include_hypernyms=False):
    """ return the sense embedding for a word most similar to its contextualized representation """ 
    sense_embeds = word_to_sense_embeddings(word, embed_map, include_hypernyms=include_hypernyms)
    if not sense_embeds:
        return vector
    return most_similar_vectors(vector, sense_embeds, n=1)[0]

def get_sense_embeddings_batch(words, words_vecs, embed_map, include_hypernyms=False, return_senses=False, map_to_bnids=True):
    """"given a list of words and a list of their vector representations, 
        returns a batch of sense embeddings related to each word.
        If no senses are found for a word, the same vector is returned
    """
    embeddings = [] 
    for word, word_vec in zip(words, words_vecs):
        sense_embeddings = word_to_sense_embeddings(word, embed_map, include_hypernyms=include_hypernyms, return_senses=return_senses, map_to_bnids=map_to_bnids)
        #most_similar returns a tensors of size (1, embed_size)
        if word_vec.shape[-1] < SENSE_EMBEDDING_DIMENSION:
                word_vec = torch.cat((word_vec, word_vec), dim=-1)
        if not sense_embeddings:
            embeddings.append(word_vec)
        else:
            if return_senses:
                embeddings.append(most_similar_senses(sense_embeddings, word_vec))
            else:
                sense_embeddings = torch.stack(sense_embeddings, dim=0)
                most_similar = most_similar_vectors(word_vec, sense_embeddings, n=1)[0].squeeze(0)
                embeddings.append(most_similar)
    if return_senses:
        return embeddings
    stacked = torch.stack(embeddings, dim=0).to(DEVICE)
    return stacked  #size (num_words, embed_dim)

def get_sense_embeddings_batches(sentences, batch, embed_map, include_hypernyms=False, indexes=None, batch_major=False):
    assert len(sentences) == batch.shape[0]
    out = []
    for i, sent in enumerate(sentences):
        reprs = batch[i]
        words = sent.strip().split(" ") if not indexes else list(indexes[i].keys())
        if indexes:
            selected_tokens = []
            words_indexes = indexes[i]
            for index in words_indexes.values():
                tokens = reprs[index]
                if len(tokens.shape) < 2:
                    tokens = tokens.unsqueeze(0)
                #print(f"tokens shape before mean: {tokens.shape}")
                tokens = torch.mean(tokens, dim=0)
                #print(f"tokens shape after mean: {tokens.shape}")
                selected_tokens.append(tokens)
            reprs = torch.stack(selected_tokens, dim=0)
            #print(f"sentence tokens shape: {reprs.shape}")
        sense_embedding = get_sense_embeddings_batch(words, reprs, embed_map, include_hypernyms)
        out.append(sense_embedding)
    if batch_major:
        return out
    return torch.stack(out, dim=0)

def build_sense_embeddings_vectors(sentences, sentences_positions, batch, embed_map):
    """
    builds sense embeddings representation for a sentence
    by substituting the word representation with the most similar sense representation
    """
    batch_sense_embeddings = []
    for sent, words_positions, vector in zip(sentences, sentences_positions, batch):
        # we take the mean of the subtokens of each word as representation
        sense_vectors = []
        for w, positions in words_positions.items():
            tokens_reprs = vector[positions]
            if len(tokens_reprs.shape) < 2:
                tokens_reprs = tokens_reprs.unsqueeze(0)
            if tokens_reprs.shape[0] > 1:
                tokens_reprs = torch.mean(tokens_reprs, dim=0, keepdim=True)
            if tokens_reprs.shape[-1] < SENSE_EMBEDDING_DIMENSION:
                tokens_reprs = torch.cat((tokens_reprs, tokens_reprs), dim=-1)
            
            sense_embeddings = word_to_sense_embeddings(w, embed_map) 
            if not sense_embeddings:
                most_similar = tokens_reprs
            else:
                most_similar = most_similar_vectors(tokens_reprs, torch.stack(sense_embeddings, dim=0), n=1)[0]
                most_similar = most_similar.unsqueeze(0)
            #most_similar = most_similar.squeeze(0)
            #print(f"Most similar shape {most_similar.shape}")
            sense_vectors.append(most_similar)
        sense_vectors = torch.stack(sense_vectors, dim=0) # dim (1, num_tokens)
        #print(f"sense vectors shape: {sense_vectors.shape}" )
        batch_sense_embeddings.append(sense_vectors)
    return batch_sense_embeddings


def reduce_dims(target_dims, tensor):
    tensor = tensor.numpy()
    tensor = norm(tensor)
    svd = TruncatedSVD(n_components=target_dims, random_state=42)
    tensor = svd.fit_transform(tensor)
    tensor = norm(tensor)
    return torch.from_numpy(tensor)

def norm(matrix):
    norm = np.sum(matrix ** 2, axis=1, keepdims=True) ** 0.5
    matrix /= norm
    return matrix


def sensekeys_to_bnids(mapping_path):
    mapping = {}
    with open(mapping_path) as f:
        for line in f:
            line = line.strip().split("\t")
            bnid, keys = line[0], line[1].split(" ")
            for key in keys:
                mapping[key] = bnid
    return mapping


####### ---------------- MODELING ----------------------------- #######




def flatten(y_true, y_pred):
    pred_flat = np.argmax(y_pred, axis=1).flatten()
    labels_flat = y_true.flatten()
    return labels_flat, pred_flat

def accuracy(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.accuracy_score(labels, preds)

def precision_score(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.precision_score(labels, preds)

def recall_score(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.recall_score(labels, preds)

def f1_score(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.f1_score(labels, preds)

def mean_squared_error(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.mean_squared_error(y_true, y_pred)

def root_mse(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return np.sqrt(self._mean_squared_error(y_true, y_pred))

def mean_squared_log_error(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.mean_squared_log_error(y_true, y_pred)

def r2(y_true, y_pred):
    labels, preds = flatten(y_true, y_pred)
    return metrics.r2_score(y_true, y_pred)

def get_accuracy_and_best_threshold_from_pr_curve(predictions, labels):
    assert 1 in labels and 0 in labels, "Some labels are not present"
    num_pos_class = sum(l for l in labels if l==1)
    num_neg_class = sum(l for l in labels if l==0)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    tp = recall * num_pos_class
    fp = (tp / precision) - tp
    tn = num_neg_class - fp
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold


class Metrics:
    """ class that acts as a manager for the metrics used in training and evaluation """
    def __init__(self, *args, mode="training"):
        """ items are instances of AverageMeter"""
        assert mode in ["training", "validation"]
        self.mode = mode
        self.metrics = []
        for m in args:
            self.metrics.append(m())

    def display_metrics(self):
        for meter in self.metrics:
            print("Current "+self.mode.upper()+" "+str(meter))

    def set_postfix(self):
        postfixes = {}
        for meter in self.metrics:
            postfixes[meter.get_name] = "{:.2f}".format(meter.avg)
        return postfixes 


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __str__(self):
        return f"average {self.name}: {self.avg}" 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def get_name(self):
        return self.name


class AccuracyMeter(AverageMeter):

    def __init__(self):
        super().__init__("accuracy")

    def __str__(self):
        return f"acc: {self.avg}"

    def update(self, preds, labels, n, **kwargs):
        self.val = accuracy(labels, preds)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

class SimilarityAccuracyMeter(AverageMeter):
    def __init__(self):
        super().__init__(name="accuracy")
        self.max_acc = 0
        self.best_threshold = -1

    def __str__(self):
        line = "accuracy: {:.2f} with threshold: {:.2f}".format(self.avg, self.best_threshold)
        return line

    def update(self, embeddings, labels, n, **kwargs):
        scores = 1-paired_cosine_distances(embeddings[0], embeddings[1])
        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=True)
        
        positive_so_far = 0
        remaining_negatives = sum(labels == 0)
        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > self.max_acc:
                self.max_acc = acc
                self.best_threshold = (rows[i][0] + rows[i+1][0]) / 2
        
        labels = [r[1] for r in rows]
        preds = [r[0] for r in rows]
        assert(len(labels)==len(preds))
        #self.max_acc, self.best_threshold = get_accuracy_and_best_threshold_from_pr_curve(preds, labels)
        thresh_preds = [1 if p >= self.best_threshold else 0 for p in preds]
        acc = metrics.accuracy_score(labels, thresh_preds)
        self.val = acc
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class SimilarityAveragePrecisionMeter(AverageMeter):
    def __init__(self):
        super().__init__(name="ap")

    def __str__(self):
        return "Average precision: {:.2f}".format(self.avg)

    def update(self, embeddings, labels, n, **kwargs):
        scores = 1-paired_cosine_distances(embeddings[0], embeddings[1])
        items = list(zip(scores, labels))
        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
        labels = [i[1] for i in items]
        scores = [i[0] for i in items]
        self.val = metrics.average_precision_score(labels, scores)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class BaseEmbedder(nn.Module):
    """ Base class for contextualized embedders """
    def __init__(self, name, retrain=False, layers=(-1, -2, -3, -4)):
        super(BaseEmbedder, self).__init__()
        self.name = name
        self.retrain = retrain
        self.layers = layers

    def _subtokenize_tokens(self, tokens):
        """ returns the tokens processed by bert tokenizer
            and a mapping to the number of their subtokens
        """
        assert isinstance(tokens, list)
        token_to_n_subtokens = {}
        tokenized = []
        for i, token in enumerate(tokens):
            t = self.tokenizer.tokenize(token)
            token_to_n_subtokens[i] = len(t)
            tokenized.extend(t)
        return tokenized, token_to_n_subtokens

    def _retokenize(self, mapping, embedding):
        """ returns the combined subtokens representazions of the tokens in the embedding for each sequence in the batch"""
        out = embedding.clone()
        assert isinstance(embedding, torch.Tensor)
        # esempio "embeddings flabbergasted" --> ['CLS', 'em', '##bed', '##ding', '##s', 'fl', '##ab', '##berg', '##ast', '##ed', 'SEP']
        tokens = self.tokenizer.convert_ids_to_tokens(mapping["input_ids"])
        assert isinstance(tokens, list)
        i = 0
        combined = []
        while(i<len(tokens)):
            token = tokens[i]
            if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                combined.append(embedding[i])
                i+=1
                continue
            if not token.startswith('##') and i < len(tokens) - 1 and tokens[i+1].startswith('##'):
                to_combine = [embedding[i]]
                j = i+1
                while(tokens[j].startswith('##')):
                    to_combine.append(tokens[j])
                    j+=1
                    i=j
                combined.append(utils.combine_tensors(to_combine))
            else:
                combined.append(embedding[i])
                i+=1
        return torch.stack(combined, dim=0)

    def forward(self, input_ids, token_type_ids, attention_mask):       
        with torch.set_grad_enabled(self.retrain):
            _, _, hidden_states, *_ = self.model.eval().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=True
            )
            hidden_states = [hidden_states[l] for l in self.layers]

        out = utils.combine_tensors(hidden_states, strategy='sum') 
        return out
           
    @property
    def device(self):
        return DEVICE


class ContextualEmbedder(BaseEmbedder):
    """ 
        Module that embeds a single sequence using BERT's 4 last layers.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(name)
        config = transformers.AutoConfig.from_pretrained(name)
        config.output_hidden_states = True
        self.hidden_size = config.hidden_size
        self.model = transformers.AutoModel.from_pretrained(name, config=config)
        self.model.to(self.device)
        for par in self.parameters():
            par.requires_grad = self.retrain
 
    @property
    def embedding_size(self):
        return self.hidden_size


class BaseModel(nn.Module):
    def __init__(
        self, 
        name, 
        hidden_size = 256, 
        num_layers=1, 
        lr=args.lr, 
        dropout_prob=args.dropout, 
        use_sense_embeddings=False,  
        train_model=False,
        normalize=False
        ):

            super(BaseModel, self).__init__()
            self.name = name
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lr = lr
            self.dropout_prob = dropout_prob
            self.use_sense_embeddings = use_sense_embeddings
            self.train_model = train_model
            self.normalize = normalize

    @property
    def model_name(self):
        return self.name


class Pooler(nn.Module):
    """Module that pools the output of another module according to different strategies """
    def __init__(self, strategy="avg", use_sense_embeddings=False, normalize=False):
        super(Pooler, self).__init__()
        assert strategy in ["avg", "cls"]
        self.strategy = strategy
        self.use_sense_embeddings = use_sense_embeddings
        self.normalize = normalize

    def forward(self, embedded_1, embedded_2, mask_1, mask_2, sentences, positions_1, positions_2):
        embedded_1 = embedded_1 * mask_1.unsqueeze(2).float() #keep only the tokens representations
        embedded_2 = embedded_2 * mask_2.unsqueeze(2).float()
        if self.use_sense_embeddings:
            #if embedded_1.shape[-1] < embeddings_config.SENSE_EMBEDDING_DIMENSION:
                #we are using ares embeddings, so duplicate the bert embedding vector size to match ares dimension
                #embedded_1_tmp = torch.cat((embedded_1, embedded_1), dim=-1)
                #embedded_2_tmp = torch.cat((embedded_2, embedded_2), dim=-1)
            sense_embeddings_1 = utils.get_sense_embeddings_batches(sentences[0], embedded_1, EMBEDDING_MAP, indexes=positions_1, batch_major=True)
            sense_embeddings_2 = utils.get_sense_embeddings_batches(sentences[1], embedded_2, EMBEDDING_MAP, indexes=positions_2, batch_major=True)
        embeddings = []
        if self.strategy == "avg":
            out_1 = embedded_1.sum(dim=1) / mask_1.sum(dim=1).view(-1, 1).float()
            out_2 = embedded_2.sum(dim=1) / mask_2.sum(dim=1).view(-1, 1).float()
            #out_1 = torch.mean(embedded_1, dim=1) #assuming shape: (batch_size, num_tokens, embed_size)
            #out_2 = torch.mean(embedded_2, dim=1)
            embeddings = [out_1, out_2]
            if self.use_sense_embeddings:
                batch_1 = []
                for word_senses_1 in sense_embeddings_1:
                    batch_1.append(torch.mean(word_senses_1, dim=0))
                batch_2 = []
                for word_senses_2 in sense_embeddings_2:
                    batch_2.append(torch.mean(word_senses_2, dim=0))
                sense_out_1 = torch.stack(batch_1, dim=0)
                sense_out_2 = torch.stack(batch_2, dim=0)
                embeddings += [sense_out_1, sense_out_2]
        elif self.strategy == "cls":
            embeddings = [embedded_1[:,0,:], embedded_2[:,0,:]] #extracting the cls token. Assuming tokens dim = 1
        if self.normalize:
            embeddings = [F.normalize(emb, p=2, dim=-1) for emb in embeddings]
        return embeddings


class BaseSiameseEmbedder(BaseModel):
    def __init__(self, pooling_strategy="avg", **kwargs):
        super().__init__(**kwargs)
        self.pooling_strategy = pooling_strategy


class SiameseSentenceEmbedder(BaseSiameseEmbedder):
    def __init__(self, loss="softmax", merge_strategy="substitute", num_classes=2, parallel_corpus=False,**kwargs):
        super().__init__(name="SiameseSentenceEmbedder", **kwargs)
        assert merge_strategy in ["substitute", "combine"]
        assert loss in ["softmax", "contrastive"]
        self.loss = loss
        self.merge_strategy = merge_strategy
        self.num_classes = num_classes
        self.parallel_corpus = parallel_corpus
        self.embedder = ContextualEmbedder(name=MODEL, retrain=self.train_model)
        self.pooler = Pooler(
            strategy=self.pooling_strategy, 
            use_sense_embeddings=self.use_sense_embeddings, 
            normalize = self.normalize
        )

        self.hidden_size = self.embedder.embedding_size*3
        
        if self.use_sense_embeddings:
            if self.merge_strategy != "combine":
                self.hidden_size = SENSE_EMBEDDING_DIMENSION*3
        
        if self.merge_strategy == "combine":
            self.sense_linear = nn.Linear(embeddings_SENSE_EMBEDDING_DIMENSION*2, self.embedder.embedding_size)

        if self.loss == "softmax":
            self.loss_fn = losses.SoftmaxLoss(self.hidden_size, self.num_classes)

        if self.loss == "contrastive":
            self.loss_fn = losses.OnlineContrastiveSimilarityLoss()

    def forward(
        self, 
        sentence_1_features, 
        sentence_2_features, 
        sentences_1, 
        sentences_2, 
        sentences_1_words_positions, 
        sentences_2_words_positions, 
        labels, 
        **kwargs
        ):
        #weights are shared, so we call only one model for both sentences
        embed_1 = self.embedder(**sentence_1_features)
        embed_2 = self.embedder(**sentence_2_features)
        
        if self.parallel_corpus:
            sentences_1 = sentences_1["tgt"]
            sentences_2 = sentences_2["tgt"]

        if self.use_sense_embeddings:
            (context_pooled_1, 
            context_pooled_2, 
            sense_pooled_1, 
            sense_pooled_2) = self.pooler(
                embed_1, 
                embed_2, 
                sentence_1_features["attention_mask"],
                sentence_2_features["attention_mask"],
                [sentences_1, sentences_2], 
                sentences_1_words_positions, 
                sentences_2_words_positions)
        else:
            (context_pooled_1, 
            context_pooled_2) = self.pooler(
                embed_1, 
                embed_2,
                sentence_1_features["attention_mask"],
                sentence_2_features["attention_mask"],
                [sentences_1, sentences_2], 
                sentences_1_words_positions, 
                sentences_2_words_positions
                )
        
        if self.loss == "contrastive":

            if self.use_sense_embeddings:
                if self.merge_strategy == "substitute":
                    embeddings, loss = self.loss_fn(sense_pooled_1, sense_pooled_2, labels)
                    return embeddings, loss
                elif self.merge_strategy == "combine":
                    embeddings_features_1 = torch.cat((context_pooled_1, sense_pooled_1), dim=-1)
                    embeddings_features_2 = torch.cat((context_pooled_2, sense_pooled_2), dim=-1)
                    embeddings, loss = self.loss_fn(embeddings_features_1, embeddings_features_2, labels)
                    return embeddings, loss
            else:
                embeddings, loss = self.loss_fn(context_pooled_1, context_pooled_2, labels)
                return embeddings, loss

        elif self.loss == "softmax":
            if self.use_sense_embeddings:

                if self.merge_strategy == "substitute":
                    diff = sense_pooled_1 - sense_pooled_2
                    class_input = torch.cat((sense_pooled_1, sense_pooled_2, diff), dim=-1)
                    logits, loss = self.loss_fn(class_input, labels)
                    return logits, loss
                
                elif self.merge_strategy == "combine":
                    sense_in = torch.cat((sense_pooled_1, sense_pooled_2), dim=-1)
                    sense_out = self.sense_linear(sense_in)
                    class_in = torch.cat((context_pooled_1, context_pooled_2, sense_out), dim=-1)
                    logits, loss = self.loss_fn(class_in, labels)
                    return logits, loss
            
            else:
                diff = context_pooled_1 - context_pooled_2
                class_in = torch.cat((context_pooled_1, context_pooled_2, diff), dim=-1)
                logits, loss = self.loss_fn(class_in, labels)
                return logits, loss



if __name__ == "__main__":

    ##### --------- TRAINING PREPARATION --------- #####

    parser = argparse.ArgumentParser()

    parser.add_argument('-ep', type=int, dest="epochs")
    parser.add_argument('--lr', type=float, dest="lr")
    parser.add_argument('--dp', type=float, dest="dropout", default=0.1)
    parser.add_argument('--bs', type=int, dest="batch_size", default=16)
    parser.add_argument('--train_path', dest="train_path", type=str)
    parser.add_argument('--valid_path', dest="valid_path", type=str)
    parser.add_argument('--langs', type=list, dest="langs")
    parser.add_argument('--merge_strategy', dest="merge_strategy", type=str)
    parser.add_argument('--train_embedder', dest="train_embedder", type=bool)
    parse.add_argument('--loss', type=str, dest="loss")
    parser.add_argument('--use_sense_embeddings', dest="use_sense_embeddings", type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True, dest="fp16")
    parser.add_argument('--parallel_dataset', type=bool, dest="parallel_dataset", default=False)
    parser.add_argument('--name', type=str, dest="name")

    args = parser.parse_args()

    processor = dataset.PawsProcessor() if not args.parallel_dataset else dataset.ParallelPawsProcessor()

    if args.parallel_dataset:
        tgt_train_paths = [f"../data/paws-x/{l}/translated_train.tsv" if l != "en" else f"../data/paws-x/{l}/train.tsv" for l in args.langs]
        tgt_valid_paths = [f"../data/paws-x/{l}/test_2k.tsv" for l in args.langs]
        train_dataset = processor.build_dataset(args.train_path, tgt_train_paths)
        valid_dataset = processor.build_dataset(args.valid_path, tgt_valid_paths)
        train_data_loader = dataset.SmartParaphraseDataloader.build_batches(train_dataset, args.batch_size, parallel_data=args.parallel_dataset)
    else:
        train_dataset = processor.build_dataset(args.train_path, args.train_path)
        valid_dataset = processor.build_dataset(args.valid_path, args.valid_path)
        train_data_loader = dataset.SmartParaphraseDataloader.build_batches(train_dataset, args.batch_size, parallel_data=args.parallel_dataset)

    model = SiameseSentenceEmbedder(
        lr=args.lr, 
        train_model=args.train_embedder,
        use_sense_embeddings=args.use_sense_embeddings,
        merge_strategy=args.merge_strategy,
        loss=args.loss
    )

    if args.loss == "contrastive":
        metrics = {"training": [SimilarityAveragePrecisionMeter], "validation": [SimilarityAveragePrecisionMeter]} 

    else:
        metrics = {"training": [AccuracyMeter], "validation": [AccuracyMeter]} 


    device = torch.device("cuda")

    num_train_steps = len(train_data_loader) * args.epochs
    num_warmup_steps = int(num_train_steps*0.1)
    learner = Learner(
        config_name=args.name, 
        model=model, 
        lr=args.lr, 
        bs=args,batch_size, 
        steps=num_train_steps, 
        warm_up_steps=num_warmup_steps, 
        device=device, 
        fp16=args.fp16, 
        metrics=metrics
    )
    trainer = Trainer(args.name, train_data_loader, valid_data_loader, args.epochs, configurations=[learner])
    trainer.execute(write_results=True)


