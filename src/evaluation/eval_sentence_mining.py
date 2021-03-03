from src.pipeline.search_pipeline import SentenceMiningPipeline
from src.models.sentence_encoder import SiameseSentenceEmbedder
from src.configurations import config
from src.modules.pooling import *
from src.models.losses import OnlineContrastiveSimilarityLoss
import random
import argparse
import torch
import transformers
from sentence_transformers import SentenceTransformer

def compare_models(queries: List[str], teacher_results: dict, student_results: dict):
    teacher_hit_sentences = set()
    student_hit_sentences = set()
    accuracy = 0
    total = 0
    for qidx in teacher_results:
        t_hits = teacher_results[qidx]
        s_hits = student_results[qidx]

        for t_hit, s_hit in zip(t_hits, s_hits):
            if t_hit == s_hit:
                accuracy += 1
                print(f"Hit for query: {queries[qidx]}")
                print(f"Sentence: {t_hit}")
                total += 1
            else:
                print(f"Results are different for query: {queries[qidx]}")
                print(f"Teacher hit: {t_hit}")
                print(f"Student hit: {s_hit}")
                total += 1
    accuracy = (accuracy / total) * 100 
    print(f"Accuracy for student model compared to teacher: {accuracy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="sentence-transformers/quora-distilbert-multilingual")
    parser.add_argument('--custom_sbert', type=str, dest="custom_sbert", default="../compression/output/distilbert-quora-multilingual-3-layers")
    parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="../training/trained_models/sencoder-dmbert-quora-to-dmbert-ted-jesc-multi")
    parser.add_argument('--perc', type=float, dest="corpus_percentage", default=0.005)
    parser.add_argument('--nq', type=int, dest="num_queries", default=10)
    parser.add_argument('--topk', type=int, dest="topk", default=1)
    parser.add_argument('--sbert', type=bool, dest="use_sbert",default=False)
    args = parser.parse_args()

    random.seed(43)

    model_config = config.ModelParameters(
        model_name = "eval_sentence_mining"
    )

    configuration = config.Configuration(
        model_parameters=model_config,
        model = args.model,
        save_path = "./results",
        batch_size = 16,
        device = torch.device(args.device),
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    )

    embedder_config = transformers.AutoConfig.from_pretrained(configuration.model)
    embedder = transformers.AutoModel.from_pretrained(configuration.model, config=embedder_config)

    if args.use_sbert:
        if args.custom_sbert is not None:
            model = SentenceTransformer(args.custom_sbert)
        else:
            model = SentenceTransformer(args.model)
    else:
        model = SiameseSentenceEmbedder.from_pretrained(args.pretrained_model_path)

    num_queries = args.num_queries

    corpus_percent = args.corpus_percentage

    document_dataset = config.load_file("../dataset/cached/jp-wikipedia-dataset")

    #TODO implement in dataloader
    all_sentences = list(document_dataset.sentences)
    random.shuffle(all_sentences)

    corpus_portion = all_sentences[:int(len(all_sentences)*corpus_percent)]
    print(f"Sentences present in the corpus: {len(corpus_portion)}")

    queries = []

    for _ in range(args.num_queries):
        q = corpus_portion[random.randint(0, len(corpus_portion))]
        queries.append(q)

    print("#### Current Queries: ####")
    for q in queries:
        print(f"query: {q}")
    print()

    pipeline = SentenceMiningPipeline(name="paraphrase_mining", corpus=corpus_portion, model=model, corpus_chunk_size=len(corpus_portion))

    results = pipeline(queries, max_num_results=args.topk)

    print("Results: ")
    for k in results:
        print(f"Best matches for query: {queries[k]}")
        print(results[k])
        print()







 



        