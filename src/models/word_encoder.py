from .modeling import BaseEncoderModel
from src.modules.pooling import *
from .losses import SoftmaxLoss
from src.modules.contextual_embedder import ContextualEmbedder
import torch
from torch import nn
from src.configurations import config as config


class WordEncoderModel(BaseEncoderModel):
    def __init__(
        self, 
        pooling_strategy: PoolingStrategy, 
        pooler: Pooler, 
        *args, 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.set_hidden_size(paraphrase=False)
        if self.params.model_parameters.use_pretrained_embeddings:
            context_pooler = WordPoolingStrategy()
            self.pooling_strategy = WordSensePoolingStrategy(
                context_pooler=context_pooler,
                return_combined=self.params.senses_as_features,
            )
        else:
            self.pooling_strategy = pooling_strategy()
        self.pooler = pooler(self.pooling_strategy, self.normalize)
        
    def forward(self, features: WordFeatures, **kwargs):
        embeddings = self.context_embedder(**features.to_dict())
        if self.params.model_parameters.use_pretrained_embeddings:
            pooled = self.pooler(
                embeddings=embeddings, 
                features=features, 
                **kwargs 
            )
        else:
            pooled = self.pooler(embeddings, features)
        return pooled

    def encode(self, features: WordFeatures, **kwargs):
        with torch.no_grad():
            encoded = self.embedder(**features.to_dict())
            pooled = self.pooler(encoded, features)
            return pooled


class WordClassifierModel(BaseEncoderModel):
    def __init__(
        self, 
        *args,
        loss: Loss = SoftmaxLoss, 
        pooler: Pooler = EmbeddingsPooler,
        pooling_strategy: PoolingStrategy = WordPoolingStrategy,
        merge_strategy: MergingStrategy = EmbeddingsCombineStrategy,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.set_hidden_size(paraphrase=True)
        self.loss = loss(self.params.model_parameters)
        self.encoder = WordEncoderModel(
            params=self.params, 
            pooler=pooler, 
            pooling_strategy=pooling_strategy
        )
        self.merge_strategy = merge_strategy()

    def forward(self, features: WordClassifierFeatures, **kwargs):
        w1_embeddings = self.encoder(features.w1_features, **kwargs)
        w2_embeddings = self.encoder(features.w2_features, **kwargs)
        merged = self.merge_strategy(features, w1_embeddings, w2_embeddings)
        return self.loss(merged, features)

    def encode(self, features: WordClassifierFeatures, **kwargs):
        encoded_1 = self.encoder.encode(features.w1_features)
        encoded_2 = self.encoder.encode(features.w2_features)
        return self.merge_strategy(features, encoded_1, encoded_2)


class GWSCModel(BaseEncoderModel):
    def __init__(self, senses_as_features=False, **kwargs):
        super().__init__(model_name=GWSCModel, **kwargs)
        self.embedder = ContextualEmbedder(model_name=config.MODEL).eval()
        self.senses_as_features = senses_as_features

    def forward(
        self, 
        sentences_1_features,
        sentences_2_features,
        w1_context1_positions,
        w2_context1_positions,
        w1_context2_positions,
        w2_context2_positions,
        lemmas_list_1,
        lemmas_list_2,
        labels
        ):

        
        embedded_1 = self.embedder(**sentences_1_features)
        embedded_2 = self.embedder(**sentences_2_features)

        w1_c1_context_embeddings = []
        w2_c1_context_embeddings = []
        w1_c2_context_embeddings = []
        w2_c2_context_embeddings = []
        for i, (w1_c1, w2_c1, w1_c2, w2_c2) in enumerate(zip(w1_context1_positions, w2_context1_positions, w1_context2_positions, w2_context2_positions)):
            w1_c1_embed = torch.mean(embedded_1[i][w1_c1], dim=0)
            w1_c1_context_embeddings.append(w1_c1_embed)
            w2_c1_embed = torch.mean(embedded_1[i][w2_c1], dim=0)
            w2_c1_context_embeddings.append(w2_c1_embed)
            w1_c2_embed = torch.mean(embedded_2[i][w1_c2], dim=0)
            w1_c2_context_embeddings.append(w1_c2_embed)
            w2_c2_embed = torch.mean(embedded_2[i][w2_c2], dim=0)
            w2_c2_context_embeddings.append(w2_c2_embed)
        w1_c1_context_embeddings = torch.stack(w1_c1_context_embeddings, dim=0)
        w2_c1_context_embeddings = torch.stack(w2_c1_context_embeddings, dim=0)
        w1_c2_context_embeddings = torch.stack(w1_c2_context_embeddings, dim=0)
        w2_c2_context_embeddings = torch.stack(w2_c2_context_embeddings, dim=0)
        embeddings = [w1_c1_context_embeddings, w2_c1_context_embeddings, w1_c2_context_embeddings, w2_c2_context_embeddings]
        sense_embeddings = []
        if self.use_pretrained_embeddings:
            if embedded_1.shape[-1] < embeddings_config.SENSE_EMBEDDING_DIMENSION:
                w1_c1_context_embeddings = torch.cat((w1_c1_context_embeddings, w1_c1_context_embeddings), dim=-1)
                w2_c1_context_embeddings = torch.cat((w2_c1_context_embeddings, w2_c1_context_embeddings), dim=-1)
                w1_c2_context_embeddings = torch.cat((w1_c2_context_embeddings, w1_c2_context_embeddings), dim=-1)
                w2_c2_context_embeddings = torch.cat((w2_c2_context_embeddings, w2_c2_context_embeddings), dim=-1)
            sense_w1_c1 = utils.get_sense_embeddings_batch(lemmas_list_1, w1_c1_context_embeddings, config.EMBEDDING_MAP)
            sense_w2_c1 = utils.get_sense_embeddings_batch(lemmas_list_1, w2_c1_context_embeddings, config.EMBEDDING_MAP)
            sense_w1_c2 = utils.get_sense_embeddings_batch(lemmas_list_2, w1_c2_context_embeddings, config.EMBEDDING_MAP)
            sense_w2_c2 = utils.get_sense_embeddings_batch(lemmas_list_2, w2_c2_context_embeddings, config.EMBEDDING_MAP)
            sense_embeddings = [sense_w1_c1, sense_w2_c1, sense_w1_c2, sense_w2_c2]

            if self.senses_as_features:
                res = []
                for context_embed, sense_embed in zip(embeddings, sense_embeddings):
                    res.append(torch.cat((context_embed, sense_embed), dim=-1))
                embeddings = res
            
            else:
                embeddings = sense_embeddings
       

        return torch.stack(embeddings, dim=0)
    
    def encode(
        self, 
        sentences_1_features,
        sentences_2_features, 
        w1_context1_positions,
        w2_context1_positions,
        w1_context2_positions,
        w2_context2_positions,
        lemmas_list_1,
        lemmas_list_2,
        labels
        ):

        return self.forward(
            sentences_1_features,
            sentences_2_features,
            w1_context1_positions,
            w2_context1_positions,
            w1_context2_positions,
            w2_context2_positions,
            lemmas_list_1,
            lemmas_list_2,
            labels
            )