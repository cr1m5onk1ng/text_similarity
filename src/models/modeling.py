import torch
from typing import Dict, Union
from torch import nn 
from src.configurations.config import Configuration
import transformers
import os


class BaseEncoderModel(nn.Module):
    """
    base class for all the encoder models
    """
    def __init__(
        self,  
        params: Configuration,
        context_embedder: nn.Module,
        normalize: bool = False
        ):
            super(BaseEncoderModel, self).__init__()
            self.params = params
            self.normalize = normalize
            self.context_embedder = context_embedder
            
    @classmethod
    def load_pretrained(cls, path, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=config)
        return cls(
            params=params,
            context_embedder=context_embedder
        )

    @property
    def model_name(self):
        return self.params.model_parameters.model_name

    def set_hidden_size(self, paraphrase=True):
        embedder_size = self.embedding_size
        pretrained_size = self.params.pretrained_embeddings_dim
        hidden_size = embedder_size if \
                    not paraphrase else \
                    embedder_size * 2
        if self.params.model_parameters.use_pretrained_embeddings:
            if self.params.senses_as_features:
                hidden_size = (embedder_size + pretrained_size) * 2
            else:
                hidden_size = embedder_size + pretrained_size
        self.params.model_parameters.hidden_size = hidden_size

    def save_pretrained(self, path):
        assert(path is not None)
        if not os.path.exists(path):
            os.makedirs(path)
        self.context_embedder.save_pretrained(path)
        self.params.tokenizer.save_pretrained(path)

    @property
    def config(self):
        return self.context_embedder.config

    @property
    def embedding_size(self):
        if "distilbert" in self.params.model:
            embed_size = self.config.dim
        else:
            embed_size = self.config.hidden_size
        if self.params.model_parameters.hidden_size is not None and self.params.model_parameters.hidden_size < embed_size:
            return self.params.embedding_Size
        return embed_size

    @property
    def params_num(self):
        return sum(param.numel() for param in self.context_embedder.parameters() if param.requires_grad)

    def forward(self):
        raise NotImplementedError()

    def encode(self):
        raise NotImplementedError()


class TransformerWrapper(BaseEncoderModel):
    def __init__(self, pooler, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = pooler
        self.loss = loss

    def forward(self, features, return_output=False, head_mask=None):
        if isinstance(features, dict):
            input_features = features
        else:
            input_features = features.to_dict()
        if head_mask is not None:
            input_features["head_mask"] = head_mask
        model_output = self.context_embedder(**input_features, output_attentions=return_output, output_hidden_states=return_output)
        if self.pooler is not None:
            pooled = self.pooler(model_output[0], features)
            output = self.loss(pooled, features)
        else:
            output = model_output[0]
        if return_output:
            return output, model_output
        return output

    @classmethod
    def load_pretrained(cls, path, pooler=None, loss=None, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        print(f"Current model config: {embedder_config}")
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        return cls(
            params=params,
            context_embedder=context_embedder,
            pooler=pooler,
            loss=loss
        )


class OnnxTransformerWrapper(BaseEncoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.params.model_parameters.hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            model_output = self.context_embedder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            model_output = self.context_embedder(input_ids=input_ids, attention_mask=attention_mask)
        return self.activation(self.linear(model_output[0]))

    @classmethod
    def load_pretrained(cls, path, pooler=None, loss=None, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        return cls(
            params=params,
            context_embedder=context_embedder,
        )
            
        




    


    
   

    






