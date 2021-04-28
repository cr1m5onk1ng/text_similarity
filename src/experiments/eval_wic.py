from src.configurations.config import WordModelConfiguration
from .wic_dataset import WicProcessor
from src.utils import utils
from src.dataset.dataset import *
from src.modules.contextual_embedder import ContextualEmbedder
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



def eval_WSD(dataset, params: WordModelConfiguration):
    """
    Evaluates the perfomance of Sense Embeddings using a 1-NN approach
    The words in context are encoded with a contextual embedder (usually BERT),
    and then a 1-NN search is performed among the sense representation of the possible
    word meaning. The resulting sense is then used for the prediction.
    """
    preds = []
    embedder = ContextualEmbedder(name=params.model).to(params.device)
    embedder.eval()
    labels = [ int(l) for l in dataset.get_labels]
    examples = dataset.get_examples
    n_correct = 0
    for i, example in enumerate(examples):
        idxs = example.get_idxs
        sent_1 = example.get_sent1.strip()
        sent_2 = example.get_sent2.strip() 
        lemma = example.get_lemma
        pos = example.pos
        w1 = sent_1.split(" ")[idxs[0]]
        w2 = sent_2.split(" ")[idxs[1]]
        sent1_tokenized = params.tokenizer.encode(sent_1)
        #print("Sent1 encoded: {}".format(sent1_tokenized))
        sent2_tokenized = params.tokenizer.encode(sent_2)[1:]
        tokenized_w1 = params.tokenizer.encode(w1)[1:-1]
        #print("Tokenized w1: {}".format(tokenized_w1))
        tokenized_w2 = params.tokenizer.encode(w2)[1:-1]
        #print("Tokenized w2: {}".format(tokenized_w2))
        w1_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w1, sent1_tokenized)
        #print("w1 token positions: {}".format(w1_token_positions))
        w2_token_positions = DataLoader.find_word_in_tokenized_sentence(tokenized_w2, sent2_tokenized)
        w2_idx1_adjusted = w2_token_positions[0] + len(sent1_tokenized)
        w2_idx2_adjusted = w2_token_positions[1] + len(sent1_tokenized)
        #print("w2 token positions: {}".format((w2_idx1_adjusted, w2_idx2_adjusted)))
        if w1_token_positions is None or w2_token_positions is None:
            raise Exception("Something went wrong, words not found in tokenized sequence!")
        positions_1 = torch.LongTensor(list(range(w1_token_positions[0],w1_token_positions[1]+1)))
        positions_2 = torch.LongTensor(list(range(w2_idx1_adjusted,w2_idx2_adjusted+1)))

        features = params.tokenizer(
            text=[[sent_1, sent_2]], 
            padding='longest',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
       
        with torch.no_grad():
            sent_embeddings = embedder(
                features["input_ids"].to(params.device), 
                features["token_type_ids"].to(config.DEVICE), 
                features["attention_mask"].to(params.device))
            
        w1_vector = sent_embeddings[0][positions_1]#skip CLS token 
        w2_vector = sent_embeddings[0][positions_2]
        w1_vector = F.normalize(w1_vector, p=2, dim=-1)
        w2_vector = F.normalize(w2_vector, p=2, dim=-1)
        
        if len(w1_vector.shape) < 2:
            w1_vector = w1_vector.unsqueeze(0)
        if len(w1_vector.shape) < 2:
            w1_vector = w1_vector.unsqueeze(0)
        w1_vector = torch.mean(w1_vector, dim=0)
        w2_vector = torch.mean(w2_vector, dim=0)

        if w1_vector.shape[-1] < params.pretrained_embeddings_dim:
            #using ARES embeddings, which are double the size of bert
            w1_vector = torch.cat((w1_vector, w1_vector), dim=-1) 
            w2_vector = torch.cat((w2_vector, w2_vector), dim=-1)
        sense_vector_map  = utils.word_to_sense_embeddings(lemma, embed_map=params.embedding_map, return_senses=True, pos=pos)
        sense_v_1, sense_1, score_1 = utils.most_similar_senses(sense_vector_map, w1_vector)[0]
        sense_v_2, sense_2, score_2 = utils.most_similar_senses(sense_vector_map, w2_vector)[0]
        print(f"Iteration: {i}")
        print(f"sent 1: {sent_1}; word 1: {w1}")
        print(f"sent 2: {sent_2}; word 1: {w2}")
        print(f"Sense found for word 1: {sense_1}. Definition: {sense_1.definition()} Sense found for word 2: {sense_2}. Definition: {sense_2.definition()}")
        print(f"label: {labels[i]}")
        print(f"Prediciton: {int(sense_1 == sense_2)}")
        label = labels[i]
        if sense_1 == sense_2:
            pred = 1
        else:
            pred = 0
            preds.append(0)
        if pred == label:
            print("correct prediciton!")
            n_correct += 1
            preds.append(pred)
        else:
            print("wrong prediction!")
        print()
    acc = n_correct/len(labels)
    print(f"final accuracy {acc}")
    return acc


if __name__ == "__main__":

    MODEL = "bert-large-cased"
    BATCH_SIZE = 4
    FP16 = True
    DEVICE = torch.device("cuda")
    NAME = "eval_wic"

    processor = WicProcessor()
    
    valid_dataset = processor.build_dataset(examples_path="../data/WiC/dev/dev.data.txt", labels_path="../data/WiC/dev/dev.gold.txt")

    res = eval_WSD(valid_dataset)
    print(res)



