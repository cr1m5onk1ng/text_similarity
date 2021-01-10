import torch
from torch.nn import functional as F
from src.models.losses import *
from src.dataset.dataset import *
import src.configurations.config as config
from src.configurations.config import ModelParameters
from src.utils import utils as utils
from typing import Union, Dict, List, Callable


class LearningStrategy(nn.Module):
    """
    Base class for tensor combining strategies
    """
    def __init__(self):
        super(LearningStrategy, self).__init__()

    def forward(self):
        raise NotImplementedError()


class PoolingStrategy(LearningStrategy):
    """
    Base class for classes that provide
    a pooling strategy for a tensor, usually
    an embedding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor):
        raise NotImplementedError()


class WordPoolingStrategy(PoolingStrategy):
    """
    The representation is pooled by extracting
    all the tokens that are part of a word in the sentence
    and taking their avarage
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: WordFeatures, **kwargs):
        word_embeddings = []
        for sen_idx, w_idxs in enumerate(features.indexes):
            curr_w_vectors = embeddings[sen_idx][w_idxs] 
            vectors_avg = torch.mean(curr_w_vectors, dim=0)
            word_embeddings.append(vectors_avg)
        return torch.stack(word_embeddings, dim=0)


class SensePoolingStrategy(PoolingStrategy):
    def __init__(self, return_combined, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_combined = return_combined

    def forward(self):
        raise NotImplementedError()


class WordSensePoolingStrategy(SensePoolingStrategy):
    def __init__(
        self, 
        context_pooler: PoolingStrategy,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.context_pooler = context_pooler
        

    def forward(self, embeddings: torch.Tensor, features: WordFeatures, **kwargs):
        embeddings = self.context_pooler(embeddings, features)
        sense_embeddings = utils.get_word_embeddings_batch(
            embeddings=embeddings, 
            embed_map=config.CONFIG.embedding_map,
            words=features.words,
            **kwargs
        )
        if self.return_combined:
            return torch.cat((embeddings, sense_embeddings), dim=-1)
        return sense_embeddings


class SiameseSensePoolingStrategy(SensePoolingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: SenseEmbeddingsFeatures, **kwargs):
        sense_embeddings = utils.get_sentence_embeddings_batch(
            embeddings=embeddings, 
            embed_map=config.CONFIG.embedding_map,
            indexes = features.tokens_indexes,
            **kwargs
        )
        if self.return_combined:
            return torch.cat((embeddings, sense_embeddings), dim=-1)
        return sense_embeddings
        

class AvgPoolingStrategy(PoolingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: EmbeddingsFeatures):
        assert len(embeddings.shape) == 3 #batch, seq_len, embed_size
        mask = features.attention_mask
        #we expand the mask to include the embed_size dimension
        mask = mask.unsqueeze(-1).expand(embeddings.size())
        #we zero out the weights corresponding to the zero positions
        # of the mask and we sum over the seq_len dimension
        out = torch.sum(embeddings * mask, 1)
        #we sum the values of the mask on the seq_len dimension
        # obtaining the number of tokens in the sequence
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        #we take the average
        embeddings = out/sum_mask 
        return embeddings


class CLSPoolingStrategy(PoolingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: EmbeddingsFeatures):
        assert len(embeddings.shape) == 3 #batch, seq_len, embed_size
        #the CLS token corresponds to the first token in the seq_len dimension
        return embeddings[:0:]


class MergingStrategy(LearningStrategy):
    """
    Base class for classes that offer functionalities for 
    merging pretrained and contextualized embeddings
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self):
        raise NotImplementedError()


class EmbeddingsCombineStrategy(MergingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features, *embeddings):
        assert embeddings is not None
        out = torch.cat(tuple(embeddings), dim=-1)
        return out


class SentenceBertCombineStrategy(MergingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features, embeddings_1, embeddings_2):
        diff = embeddings_1 - embeddings_2
        out = torch.cat((embeddings_1, embeddings_2, diff), dim=-1)
        return out


class Pooler(nn.Module):
    """Module that pools the output of another module according to different strategies """
    def __init__(
        self, 
        pooling_strategy: PoolingStrategy, 
        normalize=False
        ):
        super(Pooler, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize

    def forward(self):
        raise NotImplementedError()


class EmbeddingsPooler(Pooler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings: torch.Tensor, features: EmbeddingsFeatures, **kwargs):
        pooled = self.pooling_strategy(embeddings, features, **kwargs)
        if self.normalize:
            pooled = F.normalize(pooled, dim=-1, p=2)
        return pooled  

