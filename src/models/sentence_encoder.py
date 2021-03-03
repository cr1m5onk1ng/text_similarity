from src.utils.utils import load_file
import torch
from torch import nn
from transformers.configuration_auto import AutoConfig
from src.configurations import config as config
from .modeling import BaseEncoderModel
from .losses import *
from src.modules.pooling import *
from src.dataset.dataset import *
from typing import Union, Dict, List
import transformers
from transformers import AutoModel
import numpy
from tqdm import tqdm
import os
import json

class SiameseSentenceEmbedder(BaseEncoderModel):
    def __init__(
        self, 
        pooler: Pooler,
        pooling_strategy: PoolingStrategy, 
        *args,
        merge_strategy: MergingStrategy=None,
        loss: Loss=None,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.set_hidden_size()
        if loss is not None:
            self.loss = loss(self.params.model_parameters)
        if merge_strategy is not None:
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
     
    def forward(self, features: DataLoaderFeatures, head_mask=None):
        #weights are shared, so we call only one model for both sentences
        embed_1 = self.context_embedder(
            **features.sentence_1_features.to_dict(), 
            head_mask=head_mask)[0]
        embed_2 = self.context_embedder(
            **features.sentence_2_features.to_dict(), 
            head_mask=head_mask)[0]
        pooled_1 = self.pooler(embed_1, features.sentence_1_features)
        pooled_2 = self.pooler(embed_2, features.sentence_2_features)
        merged = self.merge_strategy(features, pooled_1, pooled_2)

        return self.loss(merged, features)

    def encode(self, features: Union[DataLoaderFeatures, EmbeddingsFeatures], output_attention=False, head_mask=None, **kwargs) -> torch.Tensor:
        output = self.context_embedder(input_ids=features["input_ids"], attention_mask=features["attention_mask"], output_attentions=output_attention, output_hidden_states=output_attention, head_mask=head_mask)
        embed = output[0]
        pooled = self.pooler(embed, features).to(self.params.device)
        if output_attention:
            return pooled, output
        return pooled

    def encode_text(self, documents: List[str], output_numpy: bool=False, eval_mode=F) -> Union[torch.Tensor, numpy.array]:
        self.to(self.params.device)
        documents = sorted(documents, key=lambda x: len(x))
        encoded_documents = []
        self.eval()
        while len(documents) > 0:
            to_take = min(self.params.batch_size, len(documents))
            select = random.randint(0, len(documents)-to_take)
            batch = documents[select:select+to_take]   
            encoded_dict = self.params.tokenizer(
                    text=batch,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=self.params.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
            )
            input_ids = encoded_dict["input_ids"].to(self.params.device)
            attention_mask = encoded_dict["attention_mask"].to(self.params.device)
            token_type_ids = encoded_dict["token_type_ids"].to(self.params.device)
            features = EmbeddingsFeatures(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            with torch.no_grad():
                embeddings = self.encode(features)
            embeddings = embeddings.detach()

            if output_numpy:
                embeddings = embeddings.cpu()

            encoded_documents.extend(embeddings)

            del documents[select:select+to_take] 
        
        if output_numpy:
            encoded_documents = numpy.asarray([embedding.numpy() for embedding in encoded_documents])
            return encoded_documents
        return torch.stack(encoded_documents)

    def set_hidden_size(self):
        embedder_size = self.context_embedder.config.hidden_size
        """
        pretrained_size = self.params.pretrained_embeddings_dim
        hidden_size = embedder_size * 3
        if self.params.model_parameters.use_pretrained_embeddings:
            if self.params.senses_as_features:
                hidden_size = (embedder_size + pretrained_size) * 2
            else:
                hidden_size = embedder_size + pretrained_size
        """
        self.params.model_parameters.hidden_size = embedder_size*3

    def load_pretrained(self, path):
        config = AutoConfig.from_pretrained(path)
        self.context_embedder = AutoModel.from_pretrained(path, config=config)

    def save_model(self, path):
        assert(path is not None)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        d = {}
        self.context_embedder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, params=None):
        config = AutoConfig.from_pretrained(path)
        context_embedder = AutoModel.from_pretrained(path, config=config)
        if params is None:
            params = torch.load(os.path.join(path, "training_params.bin"))
        return cls(
            params = params,
            context_embedder=context_embedder,
            pooling_strategy = AvgPoolingStrategy,
            pooler = EmbeddingsPooler,
        )
      