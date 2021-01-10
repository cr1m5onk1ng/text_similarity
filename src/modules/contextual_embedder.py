import torch
from torch import nn
import transformers
import src.utils.utils as utils
from src.configurations import config as config


class ContextualEmbedder(nn.Module):
    """ Base class for contextualized embedders """
    def __init__(self, name, retrain=False, layers=(-1, -2, -3, -4)):
        super(ContextualEmbedder, self).__init__()
        self.name = name
        self.retrain = retrain
        self.layers = layers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self.model_config = transformers.AutoConfig.from_pretrained(name)
        if hasattr(self.model_config, "output_hidden_states"):
            self.model_config.output_hidden_states = True
        self.hidden_size = self.model_config.hidden_size
        self.model = transformers.AutoModel.from_pretrained(name, config=self.model_config)
        self.model.to(self.device)
        for par in self.parameters():
            par.requires_grad = self.retrain

    def _subtokenize_tokens(self, tokens):
        """ returns the tokens processed by bert tokenizer
            and a mapping to the number of their subtokens
        """
        assert isinstance(tokens, list)
        token_to_n_subtokens = {}
        tokenized = []
        for i, token in enumerate(tokens):
            t = self.tokenizer.tokenize(token)
            token_to_n_subtokens[i] = len(t)
            tokenized.extend(t)
        return tokenized, token_to_n_subtokens

    def _retokenize(self, mapping, embedding):
        """ returns the combined subtokens representazions of the tokens in the embedding for each sequence in the batch"""
        out = embedding.clone()
        assert isinstance(embedding, torch.Tensor)
        # esempio "embeddings flabbergasted" --> ['CLS', 'em', '##bed', '##ding', '##s', 'fl', '##ab', '##berg', '##ast', '##ed', 'SEP']
        tokens = self.tokenizer.convert_ids_to_tokens(mapping["input_ids"])
        assert isinstance(tokens, list)
        i = 0
        combined = []
        while(i<len(tokens)):
            token = tokens[i]
            if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                combined.append(embedding[i])
                i+=1
                continue
            if not token.startswith('##') and i < len(tokens) - 1 and tokens[i+1].startswith('##'):
                to_combine = [embedding[i]]
                j = i+1
                while(tokens[j].startswith('##')):
                    to_combine.append(tokens[j])
                    j+=1
                    i=j
                combined.append(utils.combine_tensors(to_combine))
            else:
                combined.append(embedding[i])
                i+=1
        return torch.stack(combined, dim=0)

    def forward(self, input_ids, token_type_ids, attention_mask):       
        with torch.set_grad_enabled(self.retrain):
            _, _, hidden_states, *_ = self.model.forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = [hidden_states[l] for l in self.layers]

        out = utils.combine_tensors(hidden_states, strategy='sum') 
        return out
           
    @property
    def device(self):
        return config.CONFIG.device

    @property
    def embedding_size(self):
        return self.hidden_size


