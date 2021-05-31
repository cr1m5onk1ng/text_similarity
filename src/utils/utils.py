import nltk
from torch import nn
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from torch.nn import CosineSimilarity
import pickle
import os
import torch
from torch.nn import functional as F
from src.configurations import config as config


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys

def load_file(path):
    file_name = open(path, 'rb')
    d = pickle.load(file_name)
    file_name.close()
    return d

def save_file(file, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = open(os.path.join(path, name), 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def search_files(root_dir):
    for root, _, files in os.walk(root_dir):
            for name in files:
                yield os.path.join(root, name)

def count_model_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def combine_tensors(tensors, strategy='avg'):
    assert strategy in ["avg", "sum"]
    assert isinstance(tensors, list)
    if not tensors:
        raise Exception("BERT model is returning no output")
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

def remove_html_tags(text):
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def pad_punctuation(text):
    import string   
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def find_in_list(elements, target):
     for i in range(len(elements)):
         if elements[i] == target:
             return i

def index_out_of_bounds(value, *indexes):
    for index in indexes:
        for i in index:
            if i >= value:
                return True
    return False

def most_similar_vectors(vector, vectors, n=1):
    """ given a vector, finds the n most similar vectors using a similarity metric"""
    #expanding the same vector for efficiency
    expanded = vector.expand_as(vectors)
    scores = F.cosine_similarity(expanded, vectors, dim=-1)
    scores = [(i, score) for i, score in enumerate(scores)]
    ordered = list(sorted(scores, key=lambda x: x[1], reverse=True))
    res = []
    for i in range(n):
        res.append(vectors[ordered[i][0]])
    return res

def most_similar_senses(sense_vector_map, contextualized_vector, n=1):
    if not sense_vector_map:
        return (contextualized_vector, None, None)
    scores = []
    if len(contextualized_vector.shape) < 2:
        contextualized_vector = contextualized_vector.unsqueeze(0)
    scores = []
    for sense, v in sense_vector_map.items():
        if len(v.shape) < 2:
            v = v.unsqueeze(0)
        score = F.cosine_similarity(contextualized_vector, v)
        scores.append((v, sense, score))
    ordered = list(sorted(scores, key=lambda x: x[2], reverse=True))
    res = []
    for i in range(n):
        res.append(ordered[i])
    return res

def pad_tensor(tensor, padding):
    padding_tensor = torch.zeros((1, padding)).type_as(tensor)
    ret = torch.cat((tensor, padding_tensor), dim=-1)
    return ret

#map_to_bnids = True if config.pretrained_embeddings_dim == config.DIMENSIONS_MAP["ares_multi"] else False
def word_to_wn_offsets(word, include_hypernyms=False, return_mapping=False, use_sense_keys=True, map_to_bnids=False, pos=None ):
    """ given a word, returns the offsets of the synsets related to the word """
    offsets = []
    offset_to_synset = {}
    if map_to_bnids:
        bnids_map = config.bnids_map
    for synset in wn.synsets(word):
        if pos is not None:
            if synset.pos() != pos:
                continue
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
                    if map_to_bnids:
                        if key in bnids_map:
                            offset_to_synset[synset] = bnids_map[key]
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

def offsets_to_combined_tensor(offsets, embed_map, strategy='avg'):
    """ extracts the tensors associated with the specified offsets from the embedding dictionary
        and returns a vector that is the result of a combination of the vectors
    """
    assert isinstance(offsets, list)
    if not offsets:
        return None
    tensors = []
    for offset in offsets:
        if offset in embed_map:
            tensors.append(embed_map[offset])
    if tensors:
        return combine_tensors(tensors, strategy=strategy)
    else:
        return None

def word_to_sense_embeddings(word, embed_map, include_hypernyms=False, return_senses=False, map_to_bnids=False, pos=None):
    """takes a word and returns its sense embeddings representations """
    offsets = word_to_wn_offsets(word, include_hypernyms=include_hypernyms, return_mapping=return_senses, map_to_bnids=map_to_bnids, pos=pos)
    vectors = []
    sense_to_vector = {}
    if return_senses:
        for sense, offset in offsets.items():
            if offset in embed_map:
                sense_to_vector[sense] = embed_map[offset].to(config.device) 
            else:
                print("sense not found!")
    else:
        for offset in offsets:
            if offset in embed_map:
                vectors.append(embed_map[offset].to(config.device))
    if return_senses:
        return sense_to_vector
    return vectors


def get_word_embeddings_batch(embeddings, words, embed_map, include_hypernyms=False, return_senses=False, map_to_bnids=False):
    """"given a list of words and a list of their vector representations, 
        returns a batch of sense embeddings related to each word.
        If no senses are found for a word, the same vector is returned
    """
    ret_embeddings = [] 
    for word, word_vec in zip(words, embeddings):
        sense_embeddings = word_to_sense_embeddings(word, embed_map, include_hypernyms=include_hypernyms, return_senses=return_senses, map_to_bnids=map_to_bnids)
        #most_similar returns a tensors of size (1, embed_size)
        if word_vec.shape[-1] < config.pretrained_embeddings_dim:
            word_vec = torch.cat((word_vec, word_vec), dim=-1)
        if not sense_embeddings:
            if not return_senses:
                ret_embeddings.append(word_vec)
            else:
                ret_embeddings.append(most_similar_senses(sense_embeddings, word_vec))   
        else:
            #if return_senses is false, sense_embeddings is a list
            if not return_senses:
                sense_embeddings = torch.stack(sense_embeddings, dim=0)
                most_similar = most_similar_vectors(word_vec, sense_embeddings, n=1)[0].squeeze(0)
                ret_embeddings.append(most_similar)
            else:
                #if return senses is true, sense_embeddings is a dictionary
                ret_embeddings.append(most_similar_senses(sense_embeddings, word_vec))
    if return_senses:
        return ret_embeddings
    stacked = torch.stack(ret_embeddings, dim=0).to(config.device)
    return stacked  #size (num_words, embed_dim)

def get_sentence_embeddings_batch(embeddings, embed_map, indexes):
    senses_batch = []
    for i, sent_idxs in enumerate(indexes):
        sentence_embed = embeddings[i]
        sense_reprs = []
        for word, idx in sent_idxs:  
            context_embed = sentence_embed[idx]
            if context_embed.shape[-1] < config.pretrained_embeddings_dim:
                context_embed = torch.cat((context_embed, context_embed), dim=-1)
            if len(context_embed.shape) < 2:
                context_embed = context_embed.unsqueeze(0)
            context_embed = torch.mean(context_embed, dim=0)
            sense_embeddings = word_to_sense_embeddings(word, embed_map)
            if sense_embeddings:
                sense_embeddings = torch.stack(sense_embeddings, dim=0)
                one_nn = most_similar_vectors(context_embed, sense_embeddings, n=1)[0]
            else:
                one_nn = context_embed
            sense_reprs.append(one_nn)
        sense_reprs = torch.mean(torch.stack(sense_reprs, dim=0), dim=0)
        assert len(sense_reprs.shape) == 1, f"shape of each sentence: {sense_reprs.shape}"
        senses_batch.append(sense_reprs)
    return torch.stack(senses_batch, dim=0)

def word_to_combined_tensor(word, embed_map, include_hypernyms=False, strategy='avg'):
    """ given a word, returns a tensor that is a combination of his sense embeddings """
    assert isinstance(word, str)
    offsets = word_to_wn_offsets(word, include_hypernyms=include_hypernyms)
    return offsets_to_combined_tensor(offsets=offsets, embed_map=embed_map, strategy=strategy)

def load_pretrained_embeddings(embed_path, skip_first=False, embed_dim=None, reduction=False):
    """loads pretrained embeddings from file
            returns:
                a dictionary with the synset offsets as keys
                and tensors as values
    """
    embed_map = {}
    embedding_matrix = []
    if reduction:
        embedding = []
        with open(embed_path) as f:
            if skip_first:
                next(f)
            for line in f:
                parts = line.rstrip().split(" ")
                tensor = torch.tensor(list(map(float, parts[1:])))
                embedding.append(tensor) 
        embedding_matrix = torch.stack(embedding, dim=0)
        embedding_matrix = reduce_dims(embed_dim, embedding_matrix)

    with open(embed_path) as f:
        if skip_first:
            next(f)
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if reduction:
                embed_map[parts[0]] = embedding_matrix[i]
            else:
                tensor = torch.tensor(list(map(float, parts[1:])))
                if i %100000 == 0:
                    print(f"embed size: {tensor.shape}")
                embed_map[parts[0]] = tensor
    return embed_map

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



    
  