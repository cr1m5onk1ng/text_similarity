from transformers import AutoModel
from transformers import AutoTokenizer
from src.dataset.wic_dataset import *
from transformers import AutoTokenizer
from src.modules.pooling import AvgPoolingStrategy, EmbeddingsPooler, EmbeddingsSimilarityCombineStrategy, SentenceBertCombineStrategy
from src.models.sentence_encoder import SentenceTransformerWrapper, SiameseSentenceEmbedder
from src.models.losses import CosineSimilarityLoss, SoftmaxLoss
from src.models.Transformer import Transformer
from src.models.Pooling import Pooling
from src.dataset.distillation_dataset import DistillationDataset
from src.dataset.entailment_dataset import EntailmentDataset
from src.models.SentenceTransformer import SentenceTransformer
from src.utils.metrics import EmbeddingSimilarityMeter
from src.dataset.sts_dataset import StsDataset
from src.dataset.dataset import SmartParaphraseDataloader
import argparse
from src.dataset.parallel_dataset import *
from src.configurations import config
import torch
from torch.cuda import amp
from tqdm import tqdm
from heapq import heappush, heappop

def sort_by_importance(weight, bias, importance, num_instances, stride):
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)), )
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)), )
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat(sorted_bias_to_concat) if sorted_bias_to_concat is not None else None

def prune_rewire(args, model, eval_dataloader, tokenizer, use_tqdm=True):
    results = {}
    model.to(args.device)
    # get the model ffn weights and biases
    if "distilbert" in args.model:
        num_hidden_layers = model.context_embedder.config.n_layers
        intermediate_size = model.context_embedder.config.hidden_dim
        hidden_size = model.context_embedder.config.dim
        layers = model.context_embedder.base_model.transformer.layer
        num_attention_heads = model.context_embedder.config.n_heads
    else:
        num_hidden_layers = model.context_embedder.config.num_hidden_layers
        intermediate_size = model.context_embedder.config.intermediate_size
        hidden_size = model.context_embedder.config.hidden_size
        layers = model.context_embedder.base_model.encoder.layer
        num_attention_heads = model.context_embedder.config.num_attention_heads

    inter_weights = torch.zeros(num_hidden_layers, intermediate_size, hidden_size).to(args.device)
    inter_biases = torch.zeros(num_hidden_layers, intermediate_size).to(args.device)
    output_weights = torch.zeros(num_hidden_layers, hidden_size, intermediate_size).to(args.device)
    
    head_importance = torch.zeros(num_hidden_layers, num_attention_heads).to(args.device)
    ffn_importance = torch.zeros(num_hidden_layers, intermediate_size).to(args.device)

    if "distilbert" in args.model:
        for layer_num in range(num_hidden_layers):
            inter_weights[layer_num] = layers._modules[str(layer_num)].ffn.lin1.weight.detach().to(args.device)
            inter_biases[layer_num] = layers._modules[str(layer_num)].ffn.lin1.bias.detach().to(args.device)
            output_weights[layer_num] = layers._modules[str(layer_num)].ffn.lin2.weight.detach().to(args.device)
    else:
        for layer_num in range(num_hidden_layers):
            inter_weights[layer_num] = layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(args.device)
            inter_biases[layer_num] = layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(args.device)
            output_weights[layer_num] = layers._modules[str(layer_num)].output.dense.weight.detach().to(args.device)

    head_mask = torch.ones(num_hidden_layers, num_attention_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    tot_tokens = 0.0
    for batch in eval_dataloader:
        model.eval()
        batch.to(args.device)
        if args.mixed_precision:
            with amp.autocast():
                outputs = model(features=batch, return_output=False, head_mask=head_mask)
                tmp_eval_loss = outputs.loss
                logits = outputs.predictions
        else:
            outputs = model(features=batch, return_output=False, head_mask=head_mask)
            tmp_eval_loss = outputs.loss
            logits = outputs.predictions

        eval_loss += tmp_eval_loss.mean().item()

        # TODO accumulate? absolute value sum?
        tmp_eval_loss.backward()

        # collect attention confidence scores
        head_importance += head_mask.grad.abs().detach()

        # collect gradients of linear layers
        if "distilbert" in args.model:
            for layer_num in range(num_hidden_layers):
                ffn_importance[layer_num] += torch.abs(
                    torch.sum(layers._modules[str(layer_num)].ffn.lin1.weight.grad.detach()*inter_weights[layer_num], 1) 
                    + layers._modules[str(layer_num)].ffn.lin1.bias.grad.detach()*inter_biases[layer_num])
        else:
            for layer_num in range(num_hidden_layers):
                ffn_importance[layer_num] += torch.abs(
                    torch.sum(layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach()*inter_weights[layer_num], 1) 
                    + layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach()*inter_biases[layer_num])
        # TODO See if there is a better strategy
        tot_tokens += (batch.sentence_1_features.attention_mask.float().detach().sum().data + batch.sentence_2_features.attention_mask.float().detach().sum().data )

        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()

    head_importance /= tot_tokens

    # Layerwise importance normalization
    exponent = 2
    norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
    head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # rewire the network
    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    num_heads = num_attention_heads
    head_size = hidden_size / num_heads
    
    for layer_num in range(num_hidden_layers):
        # load query, key, value weights
        if "distilbert" in args.model:
            query_weight = layers._modules[str(layer_num)].attention.q_lin.weight
            query_bias = layers._modules[str(layer_num)].attention.q_lin.bias
            key_weight = layers._modules[str(layer_num)].attention.k_lin.weight
            key_bias = layers._modules[str(layer_num)].attention.k_lin.bias
            value_weight = layers._modules[str(layer_num)].attention.v_lin.weight
            value_bias = layers._modules[str(layer_num)].attention.v_lin.bias
        else:
            query_weight = layers._modules[str(layer_num)].attention.self.query.weight
            query_bias = layers._modules[str(layer_num)].attention.self.query.bias
            key_weight = layers._modules[str(layer_num)].attention.self.key.weight
            key_bias = layers._modules[str(layer_num)].attention.self.key.bias
            value_weight = layers._modules[str(layer_num)].attention.self.value.weight
            value_bias = layers._modules[str(layer_num)].attention.self.value.bias

        # sort query, key, value based on the confidence scores
        query_weight, query_bias = sort_by_importance(query_weight,
            query_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        if "distilbert" in args.model:
            layers._modules[str(layer_num)].attention.q_lin.weight = torch.nn.Parameter(query_weight)
            layers._modules[str(layer_num)].attention.q_lin.bias = torch.nn.Parameter(query_bias)
        else:
            layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
            layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = sort_by_importance(key_weight,
            key_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        if "distilbert" in args.model:
            layers._modules[str(layer_num)].attention.k_lin.weight = torch.nn.Parameter(key_weight)
            layers._modules[str(layer_num)].attention.k_lin.bias = torch.nn.Parameter(key_bias)
        else:
            layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
            layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = sort_by_importance(value_weight,
            value_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        if "distilbert" in args.model:
            layers._modules[str(layer_num)].attention.v_lin.weight = torch.nn.Parameter(value_weight)
            layers._modules[str(layer_num)].attention.v_lin.bias = torch.nn.Parameter(value_bias)
        else:
            layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
            layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

        # output matrix
        if "distilbert" in args.model:
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].attention.out_lin.weight.transpose(0, 1),
                None,
                head_importance[layer_num],
                args.target_num_heads,
                head_size)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].attention.out_lin.weight = torch.nn.Parameter(weight_sorted)
    
            weight_sorted, bias_sorted = sort_by_importance(
                layers._modules[str(layer_num)].ffn.lin1.weight,
                layers._modules[str(layer_num)].ffn.lin1.bias, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            layers._modules[str(layer_num)].ffn.lin1.weight = torch.nn.Parameter(weight_sorted)
            layers._modules[str(layer_num)].ffn.lin1.bias = torch.nn.Parameter(bias_sorted)
    
            # ffn output matrix input side
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].ffn.lin2.weight.transpose(0, 1),
                None, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].ffn.lin2.weight = torch.nn.Parameter(weight_sorted)
        else:
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].attention.output.dense.weight.transpose(0, 1),
                None,
                head_importance[layer_num],
                args.target_num_heads,
                head_size)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)

            weight_sorted, bias_sorted = sort_by_importance(
                layers._modules[str(layer_num)].intermediate.dense.weight,
                layers._modules[str(layer_num)].intermediate.dense.bias, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            layers._modules[str(layer_num)].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
            layers._modules[str(layer_num)].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)

            # ffn output matrix input side
            weight_sorted, _ = sort_by_importance(
                layers._modules[str(layer_num)].output.dense.weight.transpose(0, 1),
                None, 
                ffn_importance[layer_num],
                args.target_ffn_dim,
                1)
            weight_sorted = weight_sorted.transpose(0, 1)
            layers._modules[str(layer_num)].output.dense.weight = torch.nn.Parameter(weight_sorted)

    # save pruned model
    from pathlib import Path
    path = args.output_dir + "/pruned_" + args.config_name + "_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim))
    Path(path).mkdir(exist_ok=True)

    model.config.hidden_act = 'relu'    # use ReLU activation for the pruned models.
    if "distilbert" in args.model:
        model.config.n_heads = min([num_heads, args.target_num_heads])
        model.config.hidden_dim = layers._modules['0'].ffn.lin1.weight.size(0)
    else:
        model.config.num_attention_heads = min([num_heads, args.target_num_heads])
        model.config.intermediate_size = layers._modules['0'].intermediate.dense.weight.size(0)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    return results, preds


if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--ep', type=int, dest="epochs", default=1)
        parser.add_argument('--name', type=str, dest="config_name")
        parser.add_argument('--bs', type=int, dest="batch_size", default=16)
        parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
        parser.add_argument('--embed_dim', type=int, dest="embed_dim", default=768)
        parser.add_argument('--seq_len', type=int, dest="seq_len", default=128)
        parser.add_argument('--device', type=str, dest="device", default="cuda")
        parser.add_argument('--model', type=str, dest="model", default="sentence-transformers/quora-distilbert-multilingual")
        parser.add_argument('--pretrained', type=str, dest="pretrained_model_path", default="../compression/output/sencoder-dmbert-quora-distilled-2layers")
        parser.add_argument('--target_num_heads', type=int, dest="target_num_heads", default=12)
        parser.add_argument('--target_ffn_dim', type=int, dest="target_ffn_dim", default=600)
        parser.add_argument('--output_dir', dest="output_dir", type=str, default="./output")

        args = parser.parse_args()

        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

        model_config = config.ModelParameters(
        model_name = args.config_name,
        hidden_size = args.embed_dim,
        num_classes=3,
        freeze_weights = False,
        context_layers = (-1,)
        )

        configuration = config.ParallelConfiguration(
            model_parameters=model_config,
            model = args.model,
            sequence_max_len=args.seq_len,
            save_path = args.output_dir,
            batch_size = args.batch_size,
            epochs = args.epochs,
            device = torch.device(args.device),
            tokenizer = tokenizer,
        )

        valid_dataset = EntailmentDataset.build_dataset('../data/nli/AllNLI.tsv', mode="test")
        print()
        print(f"########## Number of examples {len(valid_dataset)} ##################")
        print()
        dataloader = SmartParaphraseDataloader.build_batches(valid_dataset, args.batch_size, mode="standard", config=configuration, sentence_pairs=False)

        sentence_model = SentenceTransformerWrapper.load_pretrained(
            path=args.model,
            params=configuration,
            merge_strategy=SentenceBertCombineStrategy(),
            loss = SoftmaxLoss(params=configuration)
        )

        prune_rewire(args, sentence_model, dataloader, tokenizer)
