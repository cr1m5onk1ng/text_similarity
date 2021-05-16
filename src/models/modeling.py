from src.dataset.dataset import DataLoaderFeatures
import torch
from typing import Dict, Optional, Union
from torch import nn
from src.configurations.config import Configuration
from src.modules.modules import AvgPoolingStrategy, BertPoolingStrategy, SoftmaxLoss
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
        input_dict: bool = False,
        normalize: bool = False
        ):
            super(BaseEncoderModel, self).__init__()
            self.params = params
            self.normalize = normalize
            self.input_dict = input_dict
            self.context_embedder = context_embedder
            
    @classmethod
    def from_pretrained(cls, path, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=config)
        return cls(
            params=params,
            context_embedder=context_embedder
        )

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
        config_path = os.path.join(path, "model_config.bin")
        torch.save(self.params, config_path)

    @property
    def model_name(self):
        return self.params.model_parameters.model_name

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
    """
    A simple wrapper around Huggingface pretrained models.
    :param pooler: A pooler module that reduces the tokens dimension to a fixed sized, usually by taking the CLS token representation
    :param loss: A module that represents the loss applied during training for various downstream tasks
    """
    def __init__(self, pooler: nn.Module, loss: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooler = pooler
        self.loss = loss

    def forward(self, features: DataLoaderFeatures, return_output: bool=False, head_mask: torch.Tensor=None, **kwargs):
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

    def save_pretrained(self, path: str):
        super().save_pretrained(path)
        save_dict = {}
        if hasattr(self, "pooler"):
            save_dict["pooler"] = self.pooler.state_dict()
        if hasattr(self, "loss"):
            save_dict["loss"] = self.loss.state_dict()
        torch.save(save_dict, os.path.join(path, "modules.bin"))


    @classmethod
    def from_pretrained(cls, path, pooler=None, loss=None, params=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        checkpoint = torch.load(os.path.join(path, "modules.bin"))
        if pooler is not None:
            if "pooler" in checkpoint:
                pooler.load_state_dict(checkpoint["pooler"])
        if loss is not None:
            if "loss" in checkpoint:
                loss.load_state_dict(checkpoint["loss"])
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        return cls(
            params=params,
            context_embedder=context_embedder,
            pooler=pooler,
            loss=loss
        )


class OnnxTransformerWrapper(BaseEncoderModel):
    """
    A simple wrapper around Huggingface pretrained models whose purpose is to be optimized for inference. As such,
    the forward function doesnt contain any branching.
    :param pooler: A pooler module that reduces the tokens dimension to a fixed sized, usually by taking the CLS token representation
    :param output: A module that represents the output layer applied during for a downstream tasks (usually a simple linear layer or MLP)
    """
    def __init__(self, *args, pooler: nn.Module = None, output: nn.Module = None, **kwargs):
        super().__init__(*args, input_dict=True, **kwargs)
        self.pooler = pooler
        self.output = output
        if self.pooler is None:
            self.pooler = BertPoolingStrategy(self.params)
        if self.output is None:
            self.output = nn.Linear(self.params.model_parameters.hidden_size, self.params.model_parameters.num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        model_output = self.context_embedder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(model_output[0], None)
        return self.output(pooled)

    def save_pretrained(self, path: str):
        super().save_pretrained(path)
        torch.save(
            {
                "pooler": self.pooler.state_dict(),
                "output": self.output.state_dict(),
            }, 
            os.path.join(path, "modules.bin")
        )

    @classmethod
    def from_pretrained(cls, path: str, pooler: nn.Module=None, output: nn.Module=None, params: Configuration=None):
        if params is None:
            params = torch.load(os.path.join(path, "model_config.bin"))
        checkpoint = torch.load(os.path.join(path, "modules.bin"))
        if pooler is None:
            pooler = BertPoolingStrategy(params)
            if "pooler" in checkpoint:
                pooler.load_state_dict(checkpoint["pooler"])
        if output is None:
            output = nn.Linear(params.model_parameters.hidden_size, params.model_parameters.num_classes)
            if "output" in checkpoint:
                output.load_state_dict(checkpoint["output"])
        embedder_config = transformers.AutoConfig.from_pretrained(path)
        context_embedder = transformers.AutoModel.from_pretrained(path, config=embedder_config)
        return cls(
            params=params,
            context_embedder=context_embedder,
            pooler=pooler,
            output=output
        )
            
        




    


    
   

    






