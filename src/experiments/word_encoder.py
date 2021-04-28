from .modeling import BaseEncoderModel
from src.modules.modules import *
from src.modules.contextual_embedder import ContextualEmbedder
import torch
from src.configurations import config as config


class WordEncoderModel(BaseEncoderModel):
    """
    Model that produces word-level embeddings.
    Primarly used for word-level sense disambiguation and text similarity tasks
    """
    def __init__(
        self, 
        pooling_strategy: PoolingStrategy, 
        *args, 
        merge_strategy: MergingStrategy = None,
        loss: Loss = None,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.set_hidden_size(paraphrase=False)
        if self.params.model_parameters.use_pretrained_embeddings:
            context_pooler = WordPoolingStrategy()
            self.pooling_strategy = WordSensePoolingStrategy(
                params=self.params,
                context_pooler=context_pooler,
                return_combined=self.params.senses_as_features,
            )
        else:
            self.pooling_strategy = pooling_strategy(params=self.params)
        self.merge_strategy = merge_strategy
        self.loss = loss(params=self.params)
        
    def forward(self, features: WordFeatures, **kwargs):
        embeddings = self.context_embedder(**features.embeddings_features.to_dict())[0]
        if self.params.model_parameters.use_pretrained_embeddings:
            pooled = self.pooling_strategy(
                embeddings=embeddings, 
                features=features.embeddings_features, 
                **kwargs 
            )
        else:
            pooled = self.pooling_strategy(embeddings, features.embeddings_features)
        return self.loss(pooled, features)

    def encode(self, features: WordFeatures, **kwargs):
        with torch.no_grad():
            encoded = self.context_embedder(**features.to_dict())
            pooled = self.pooling_strategy(encoded, features.embeddings_features)
            return pooled


class GWSCModel(BaseEncoderModel):
    """
    Model specific to the task Graded Word Similarity in Context, part of
    the SemEval 2020 tasks. This is a word-level text similarity task
    that heavily relises on the ability of the model to perform Word Sense Disambiguation
    """
    def __init__(self, params: config.WordModelConfiguration, *args, senses_as_features=False, **kwargs):
        super().__init__(*args, params=params, **kwargs)
        self.embedder = ContextualEmbedder(model_name=self.params.model)
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
        self.embedder.eval()
        
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
            if embedded_1.shape[-1] < self.params.model_parameters.sense:
                w1_c1_context_embeddings = torch.cat((w1_c1_context_embeddings, w1_c1_context_embeddings), dim=-1)
                w2_c1_context_embeddings = torch.cat((w2_c1_context_embeddings, w2_c1_context_embeddings), dim=-1)
                w1_c2_context_embeddings = torch.cat((w1_c2_context_embeddings, w1_c2_context_embeddings), dim=-1)
                w2_c2_context_embeddings = torch.cat((w2_c2_context_embeddings, w2_c2_context_embeddings), dim=-1)
            sense_w1_c1 = utils.get_sense_embeddings_batch(lemmas_list_1, w1_c1_context_embeddings, self.params.embedding_map)
            sense_w2_c1 = utils.get_sense_embeddings_batch(lemmas_list_1, w2_c1_context_embeddings, self.params.embedding_map)
            sense_w1_c2 = utils.get_sense_embeddings_batch(lemmas_list_2, w1_c2_context_embeddings, self.params.embedding_map)
            sense_w2_c2 = utils.get_sense_embeddings_batch(lemmas_list_2, w2_c2_context_embeddings, self.params.embedding_map)
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