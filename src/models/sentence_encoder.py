from tqdm.std import trange
from src.utils.utils import load_file
import torch
from torch import nn
from transformers import AutoConfig
from src.configurations import config as config
from .modeling import BaseEncoderModel
from .losses import *
from src.modules.pooling import *
from src.dataset.dataset import *
from typing import Union, Dict, List
from transformers import AutoModel
import transformers
import numpy as np
from tqdm import tqdm
import os


class OnnxSentenceTransformerWrapper(BaseEncoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = nn.Linear(self.embedding_size, self.params.model_parameters.hidden_size)

    def forward(self, input_ids, attention_mask):
        token_embeddings = self.context_embedder(input_ids=input_ids, attention_mask=attention_mask)[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return self.projection(pooled)

    @classmethod
    def load_pretrained(cls, path, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        context_embedder = transformers.AutoModel.from_pretrained(path)
        return cls(
            params=params,
            context_embedder=context_embedder
        )


class SentenceTransformerWrapper(BaseEncoderModel):
    def __init__(self, pooler: Pooler, merge_strategy: MergingStrategy, loss: Loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = pooler
        if merge_strategy is not None:
            self.merge_strategy = merge_strategy
        if loss is not None:
            self.loss = loss
        if self.params.model_parameters.hidden_size < self.embedding_size:
            self.projection = nn.Linear(self.embedding_size, self.params.model_parameters.hidden_size, bias=False)

    def forward(self, features, parallel_mode=True, return_output=False, head_mask=None):
        if parallel_mode:
            features_1 = features.sentence_1_features.to_dict()
            features_2 = features.sentence_2_features.to_dict()
            if head_mask is not None:
                features_1['head_mask'] = head_mask
                features_2['head_mask'] = head_mask
            embed_features_1 = self.context_embedder(**features_1)[0]
            embed_features_2 = self.context_embedder(**features_2)[0]
            embed_1 = self.pooler(embed_features_1, features.sentence_1_features)
            embed_2 = self.pooler(embed_features_2, features.sentence_2_features)
            merged = self.merge_strategy(features, embed_1, embed_2)
        else:
            assert isinstance(features, EmbeddingsFeatures)
            input_features = features.to_dict()
            if head_mask is not None:
                input_features['head_mask'] = head_mask
            model_output = self.context_embedder(**input_features, output_attentions=return_output, output_hidden_states=return_output)
            pooled = self.pooler(model_output[0], features)
            merged = pooled
        if hasattr(self, "projection"):
            merged = self.projection(pooled)
        output = self.loss(merged, features)
        if not parallel_mode:
            if return_output:
               return output, model_output
        return output

    def encode(self, features: EmbeddingsFeatures, return_output=False, **kwargs) -> torch.Tensor:
        output = self.context_embedder(**features.to_dict(), output_attentions=return_output, output_hidden_states=return_output)
        pooled = self.pooler(output[0], features)
        if return_output:
            return pooled, output
        return pooled

    def encode_text(self, documents: List[str], output_np: bool=False) -> Union[torch.Tensor, np.array]:
        self.to(self.params.device)
        length_sorted_idx = np.argsort([len(sen) for sen in documents])
        documents = [documents[idx] for idx in length_sorted_idx]
        encoded_documents = []
        self.eval()
        for start_index in trange(0, len(documents), self.params.batch_size):
            sentences_batch = documents[start_index:start_index+self.params.batch_size]   
            encoded_dict = self.params.tokenizer(
                    text=sentences_batch,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=self.params.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors='pt'
            )
            input_ids = encoded_dict["input_ids"].to(self.params.device)
            attention_mask = encoded_dict["attention_mask"].to(self.params.device)
            features = EmbeddingsFeatures(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
            )
            with torch.no_grad():
                embeddings = self.encode(features, parallel_mode=False)
            embeddings = embeddings.detach()

            if output_np:
                embeddings = embeddings.cpu()

            encoded_documents.extend(embeddings)
        encoded_documents = [encoded_documents[idx] for idx in np.argsort(length_sorted_idx)]
        
        if output_np:
            encoded_documents = np.asarray([embedding.numpy() for embedding in encoded_documents])
            return encoded_documents
        return torch.stack(encoded_documents)

    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        config_path = os.path.join(path, "model_config.bin")
        torch.save(self.params, config_path)
        self.context_embedder.save_pretrained(path)
        self.context_embedder.config.save_pretrained(path)
    
    def get_sentence_embedding_dimension(self):
        return self.context_embedder.config.hidden_size

    @classmethod
    def load_pretrained(cls, path, merge_strategy=None, loss=None, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        pooler = AvgPoolingStrategy()
        return cls(
            params=params,
            context_embedder=context_embedder,
            merge_strategy=merge_strategy,
            pooler=pooler,
            loss=loss
        )


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
            self.loss = loss(self.params)
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
     
    def forward(self, features: Union[DataLoaderFeatures, EmbeddingsFeatures], eval_mode=False, head_mask=None):
        #weights are shared, so we call only one model for both sentences
        if eval_mode:
            embedding = self.context_embedder(**features.sentence_2_features.to_dict(), head_mask=head_mask)[0]
            pooled = self.pooler(embedding, features.sentence_2_features, head_mask=head_mask)
            merged = self.merge_strategy(features, pooled)
        else:
            embed_1 = self.context_embedder(
                **features.sentence_1_features.to_dict(), head_mask=head_mask)[0]
            embed_2 = self.context_embedder(
                **features.sentence_2_features.to_dict(), head_mask=head_mask)[0]
            pooled_1 = self.pooler(embed_1, features.sentence_1_features)
            pooled_2 = self.pooler(embed_2, features.sentence_2_features)
            merged = self.merge_strategy(features, pooled_1, pooled_2)

        return self.loss(merged, features)

    def encode(self, features: Union[DataLoaderFeatures, EmbeddingsFeatures], output_attention=False, head_mask=None, **kwargs) -> torch.Tensor:
        output = self.context_embedder(input_ids=features.input_ids, attention_mask=features.attention_mask, output_attentions=output_attention, output_hidden_states=output_attention, head_mask=head_mask)
        embed = output[0]
        pooled = self.pooler(embed, features).to(self.params.device)
        if output_attention:
            return pooled, output
        return pooled

    def encode_text(self, documents: List[str], output_np: bool=False) -> Union[torch.Tensor, np.array]:
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
            features = EmbeddingsFeatures(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
            )
            with torch.no_grad():
                embeddings = self.encode(features)
            embeddings = embeddings.detach()

            if output_np:
                embeddings = embeddings.cpu()

            encoded_documents.extend(embeddings)

            del documents[select:select+to_take] 
        
        if output_np:
            encoded_documents = np.asarray([embedding.np() for embedding in encoded_documents])
            return encoded_documents
        return torch.stack(encoded_documents)

    def set_hidden_size(self):
        embedder_size = self.context_embedder.config.hidden_size
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
      