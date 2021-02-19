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
     
    def forward(self, features: SiameseDataLoaderFeatures, head_mask=None):
        #weights are shared, so we call only one model for both sentences
        embed_1 = self.context_embedder(
            **features.sentence_1_features.to_dict(), 
            head_mask=head_mask,
            output_attentions=self.params.model_parameters.output_attention)[0]
        embed_2 = self.context_embedder(
            **features.sentence_2_features.to_dict(), 
            head_mask=head_mask,
            output_attentions=self.params.model_parameters.output_attention)[0]

        pooled_1 = self.pooler(embed_1, features.sentence_1_features)
        pooled_2 = self.pooler(embed_2, features.sentence_2_features)
        
        merged = self.merge_strategy(features, pooled_1, pooled_2)

        return self.loss(merged, features)

    def encode(self, features: DataLoaderFeatures, **kwargs):
        with torch.no_grad():
            embed = self.embedder(**features.embeddings_features.to_dict())[0]
            pooled = self.pooler(embed, features.embeddings_features)
        return pooled

    def set_hidden_size(self):
        embedder_size = self.context_embedder.config.hidden_size
        pretrained_size = self.params.pretrained_embeddings_dim
        hidden_size = embedder_size * 3
        if self.params.model_parameters.use_pretrained_embeddings:
            if self.params.senses_as_features:
                hidden_size = (embedder_size + pretrained_size) * 2
            else:
                hidden_size = embedder_size + pretrained_size
        self.params.model_parameters.hidden_size = hidden_size

    def load_pretrained(self, path):
        checkpoint = torch.load(path)
        self.context_embedder.load_state_dict(checkpoint["embedder_state_dict"], strict=True)
        self.pooler.load_state_dict(checkpoint['pooler_state_dict'], strict=True)