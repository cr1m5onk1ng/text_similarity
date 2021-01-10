
from src.modules.transformers_bert import BertModel
from transformers import BertTokenizer
from transformers import BertConfig
import torch


if __name__ == '__main__':

    config = BertConfig()
    config.max_position_embeddings = 1024

    model = BertModel(config=config)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    sentence = 'Hi how are you today?'

    d = tokenizer(sentence, return_attention_mask=True, return_tensors='pt')
    ids = d["input_ids"]
    mask = d["attention_mask"]
    out = model(input_ids=ids, attention_mask=mask)

    print(out[0])



