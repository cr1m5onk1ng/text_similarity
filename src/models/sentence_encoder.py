import torch
from torch import nn
from src.configurations import config as config
from .modeling import BaseEncoderModel
from .losses import *
from src.modules.pooling import *
from src.dataset.dataset import *
from typing import Union, Dict, List


class SiameseSentenceEmbedder(BaseEncoderModel):
    def __init__(
        self, 
        pooler: Pooler,
        pooling_strategy: PoolingStrategy, 
        merge_strategy: MergingStrategy,
        loss: Loss,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.set_hidden_size()
        self.loss = loss(self.params.model_parameters)
        self.merge_strategy = merge_strategy()
        if self.params.model_parameters.use_pretrained_embeddings:
            self.pooling_strategy = SiameseSensePoolingStrategy(
                return_combined=self.params.senses_as_features   
            )
        else:
            self.pooling_strategy = pooling_strategy()

        self.pooler = pooler(
            pooling_strategy = self.pooling_strategy, 
            normalize = self.normalize
        )
     
    def forward(self, features: SiameseDataLoaderFeatures):
        #weights are shared, so we call only one model for both sentences
        embed_1 = self.context_embedder(**features.sentence_1_features.to_dict())
        embed_2 = self.context_embedder(**features.sentence_2_features.to_dict())

        pooled_1 = self.pooler(embed_1, features.sentence_1_features)
        pooled_2 = self.pooler(embed_2, features.sentence_2_features)
        
        merged = self.merge_strategy(features, pooled_1, pooled_2)

        return self.loss(merged, features)

    def encode(self, features: SiameseDataLoaderFeatures, **kwargs):
        with torch.no_grad():
            embed_1 = self.embedder(**features.sentence_1_features.to_dict())
            embed_2 = self.embedder(**features.sentence_2_features.to_dict())

            pooled_1 = self.pooler(embed_1, features.sentence_1_features)
            pooled_2 = self.pooler(embed_2, features.sentence_2_features)

            return self.merge_strategy(features, pooled_1, pooled_2)

    def set_hidden_size(self):
        embedder_size = self.context_embedder.embedding_size
        pretrained_size = self.params.pretrained_embeddings_dim
        hidden_size = embedder_size * 3
        if self.params.model_parameters.use_pretrained_embeddings:
            if self.params.senses_as_features:
                hidden_size = (embedder_size + pretrained_size) * 2
            else:
                hidden_size = embedder_size + pretrained_size
        self.params.model_parameters.hidden_size = hidden_size