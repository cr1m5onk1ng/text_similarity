import torch
from typing import Dict, Union
from torch import nn 
from torch.nn import functional as F
from src.modules.contextual_embedder import ContextualEmbedder
import src.configurations.config as config
from src.configurations.config import ModelParameters, Configuration
from src.models.losses import *
import src.utils.utils as utils
from src.utils import activations
from src.modules.pooling import *
from abc import ABC, abstractmethod
import transformers


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
        if isinstance(self.context_embedder, SentenceTransformer):
            embedder_size = self.context_embedder.get_sentence_embedding_dimension()
        else:
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


class EmbeddingsAlignmentModel(BaseEncoderModel):
    def __init__(self, model_path):
        super().__init__()

        #load the model
        self.model = SiameseEmbedderModel()
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, src_lang_features, target_lang_features, **kwargs):

        src_embeddings = self.model(**src_lang_features)
        target_embeddings = self.model(**target_lang_features)

        mse = ((src_embeddings - target_embeddings)**2).mean()
        mse*=100

        return torch.stack(src_embeddings, target_embeddings, dim=0), mse

        
class MBERTClassifier(BaseEncoderModel):
    def __init__(self, num_classes=2, senses_as_features=False, strategy="avg", **kwargs):
        super().__init__(model_name="mBERT", **kwargs)
        self.num_classes = num_classes
        self.embedder = ContextualEmbedder(model_name=config.MODEL, retrain=self.freeze_weights) 
        self.embeddings_size = self.embedder.embedding_size 
        self.senses_as_features = senses_as_features
        self.strategy = strategy
        self.pooler = AvgPooler(
            use_pretrained_embeddings=self.use_pretrained_embeddings, 
            embeddings_size=self.embeddings_size, 
            senses_as_features=senses_as_features, 
            strategy=self.strategy,
            normalize=self.normalize)
        #self.dropout = nn.Dropout(self.dropout_prob)
        if self.use_pretrained_embeddings:
            if self.senses_as_features:
                self.hidden_size = self.embeddings_size + embeddings_config.SENSE_EMBEDDING_DIMENSION
            else:
                self.hidden_size = embeddings_config.SENSE_EMBEDDING_DIMENSION 
        else:
            self.hidden_size = self.embedder.embedding_size 
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, sentences_words_positions, labels, **kwargs):
        embed = self.embedder(input_ids, token_type_ids, attention_mask)
        
        pooled = self.pooler(
            embed,
            attention_mask,
            positions=sentences_words_positions)

        #out = self.dropout(pooled)
    
        logits = self.classifier(pooled)
        loss = self.loss(logits.view(-1, self.num_classes), labels)
        return logits, loss

    def encode(self, sentence_1_features, sentence_2_features, **kwargs):
        self.use_pretrained_embeddings = False
        self.pooler.use_pretrained_embeddings = False

        embed_1 = self.embedder(**sentence_1_features)
        
        pooled_1 = self.pooler(
            embed_1,
            sentence_1_features["attention_mask"],
            positions=None
        )

        embed_2 = self.embedder(**sentence_2_features)
        
        pooled_2 = self.pooler(
            embed_2,
            sentence_2_features["attention_mask"],
            positions=None
        )
    
        return torch.stack([pooled_1, pooled_2], dim=0)




if __name__ == '__main__':
    from src.dataset.dataset import *

    model = SiameseEmbedderModel(use_pretrained_embeddings=True, loss="contrastive").to(config.DEVICE)   

    dataset = utils.load_file("../data/cached/paswx_en")
    dataloader = SmartParaphraseDataloader.build_batches(dataset, 16)
    out = model(**dataloader.batches[0])
    print(out[0].shape)



    


    
   

    






