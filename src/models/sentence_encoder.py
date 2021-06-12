from tqdm.std import trange
from src.utils.utils import load_file
import torch
from torch import nn
from transformers import AutoConfig
from src.configurations import config as config
from src.modules.modules import *
from typing import Union, Dict, List
from src.models.modeling import BaseEncoderModel
from transformers import AutoModel
import transformers
import numpy as np
import os



class OnnxSentenceTransformerWrapper(BaseEncoderModel):
    """
    A wrapper around Huggingface pretrained model,following the SBERT Bi-Encoder architecutre, whose purpose is to be
    optimized for inference. In this case, the forward function doesnt contain any branching and the pooler behaviour is
    fixed to average pooling.

    :param projection_dim: The dimension of the projection matrix, applied in the case we want to reduce the size of the output embeddings 
    """
    def __init__(self, *args, projection: nn.Module=None,  **kwargs):
        super().__init__(*args, **kwargs)
        if projection is None:
            self.projection = nn.Identity()
        else:
            self.projection = projection

    def forward(self, input_ids, attention_mask, **kwargs):
        token_embeddings = self.context_embedder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]
        token_embeddings = self.projection(token_embeddings)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled

    @classmethod
    def from_pretrained(cls, path, projection: nn.Module=None, params: Configuration=None):
        config_path = os.path.join(path, "model_config.bin")
        if not os.path.exists(config_path):
            assert params is not None, "Parameters not found, need to pass model parameters for the model to work"
        else:
            params = torch.load(config_path)
        config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=config)
        checkpoint_path = os.path.join(path, "modules.bin")
        checkpoint = None
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        if projection is not None:
            if checkpoint is not None:
                if "projection" in checkpoint:
                    projection.load_state_dict(checkpoint["projection"])
        return cls(
            params=params,
            context_embedder=context_embedder,
            projection=projection
        )

    def save_pretrained(self, path):
        super().save_pretrained(path)
        self.context_embedder.save_pretrained(path)
        self.context_embedder.config.save_pretrained(path)
        self.params.tokenizer.save_pretrained(path)
        torch.save({"projection": self.projection.state_dict()}, os.path.join(path, "modules.bin"))


class SentenceTransformerWrapper(BaseEncoderModel):
    """
    A wrapper around Huggingface pretrained model,following the SBERT Bi-Encoder architecutre
    :param pooler: A pooler module that reduces the tokens dimension to a fixed sized, usually by taking the tokens dimension average
    :param merge_strategy: A module that combines the output of the two sentences representations following a specific strategy
    :param loss: A module that represents the loss applied during training for various downstream tasks
    :param parallel_mode: Whether or not the model forward function is behaving like a bi-encoder or is outputting embeddings a (batch of) sentences
    :param projection_dim: The dimension of the projection matrix, applied in the case we want to reduce the size of the outputted embeddings 
    """
    def __init__(
        self, 
        pooler: PoolingStrategy, 
        merge_strategy: MergingStrategy, 
        loss: Loss, 
        *args, 
        parallel_mode: bool=True,
        projection: nn.Module=None, 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = pooler
        self.merge_strategy = merge_strategy
        self.loss = loss
        self.parallel_mode = parallel_mode
        self.projection = projection
        if self.projection is None:
            self.projection = nn.Identity()


    def forward(self, features, return_output=False, head_mask=None):
        if self.parallel_mode:
            features_1 = features.sentence_1_features.to_dict()
            features_2 = features.sentence_2_features.to_dict()
            if head_mask is not None:
                features_1['head_mask'] = head_mask
                features_2['head_mask'] = head_mask
            embed_1 = self.context_embedder(**features_1)[0]
            embed_2 = self.context_embedder(**features_2)[0]
            embed_1 = self.pooler(embed_1, features.sentence_1_features)
            embed_2 = self.pooler(embed_2, features.sentence_2_features)
            #merged = self.merge_strategy(features, embed_1, embed_2)
            diff = torch.abs(embed_1 - embed_2)
            merged = torch.cat((embed_1, embed_2, diff), dim=-1)
        else:
            input_features = features.to_dict()
            if head_mask is not None:
                input_features['head_mask'] = head_mask
            model_output = self.context_embedder(**input_features, output_attentions=return_output, output_hidden_states=return_output)
            if hasattr(self, "projection"):
                pooled = self.pooler(model_output[0], features)
                pooled = self.projection(pooled)
            else:
                pooled = self.pooler(model_output[0], features)
            merged = pooled
        if hasattr(self, "projection"):
            merged = self.projection(merged)
        output = self.loss(merged, features)
        if not self.parallel_mode:
            if return_output:
               return output, model_output
        return output

    def encode(self, documents: List[str], output_np: bool=False) -> Union[torch.Tensor, np.array]:
        return self.encode_text(documents, output_np)

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
    
    def get_sentence_embedding_dimension(self):
        return self.context_embedder.config.hidden_size

    def save_pretrained(self, path):
        super().save_pretrained(path)
        save_dict = {}
        if hasattr(self, "pooler"):
            save_dict["pooler"] = self.pooler.state_dict()
        if hasattr(self, "loss"):
            save_dict["loss"] = self.loss.state_dict()
        torch.save(save_dict, os.path.join(path, "modules.bin"))

    @classmethod
    def from_pretrained(cls, path, pooler=None, merge_strategy=None, loss=None, params=None, parallel_mode=True):
        config_path = os.path.join(path, "model_config.bin")
        if not os.path.exists(config_path):
            assert params is not None, "Parameters not found, need to pass model parameters for the model to work"
        else:
            params = torch.load(config_path)
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        checkpoint_path = os.path.join(path, "modules.bin")
        checkpoint = None
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        if pooler is None:
            pooler = AvgPoolingStrategy()
            if checkpoint is not None:
                if "pooler" in checkpoint:
                    pooler.load_state_dict(checkpoint["pooler"])
                
        if loss is not None:
            if checkpoint is not None:
                if "loss" in checkpoint:
                    loss.load_state_dict(checkpoint["loss"])
        
        return cls(
            params=params,
            context_embedder=context_embedder,
            pooler=pooler,
            merge_strategy=merge_strategy,
            loss=loss
        )

