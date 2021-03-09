from src.pipeline.search_pipeline import APISearchPipeline, SemanticSearchPipeline, SentenceMiningPipeline
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.configurations import config
from src.modules.pooling import *
import random
import argparse
import torch
import transformers
from sentence_transformers import SentenceTransformer

def compare_models(queries: List[str], teacher_results: dict, student_results: dict):
    accuracy = 0
    total = 0
    for qidx in teacher_results:
        t_hits = teacher_results[qidx]
        s_hits = student_results[qidx]
        for i in range(len(t_hits)):
            print(f"Teacher result {i+1} for query {qidx+1}: {t_hits[i]}")
            print(f"Student result {i+1} for query {qidx+1}: {s_hits[i]}")
        print()
        print(f"Hits for query: {queries[qidx]}")
        print()
        for i, s_hit in enumerate(s_hits):
            if s_hit in t_hits:
                accuracy += 1  
                print(f"Sentence {i+1}: {s_hit}")
                total += 1
            else:
                print(f"Results for the query are different for student hit number: {i+1}")
                print(f"Student hit {i+1}: {s_hit}")
                total += 1
        print()
    accuracy = (accuracy / total) * 100 
    print(f"Accuracy for student model compared to teacher: {accuracy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--student_model', type=str, dest="student_model", default="sentence-transformers/quora-distilbert-multilingual")
    parser.add_argument('--teacher_model', type=str, dest="teacher_model", default="sentence-transformers/quora-distilbert-multilingual")
    parser.add_argument('--perc', type=float, dest="corpus_percentage", default=0.01)
    parser.add_argument('--nq', type=int, dest="num_queries", default=10)
    parser.add_argument('--topk', type=int, dest="topk", default=3)
    args = parser.parse_args()

    random.seed(43)

    model_config = config.ModelParameters(
        model_name = "eval_sentence_mining"
    )

    configuration_teacher = config.Configuration(
        model_parameters=model_config,
        model = args.teacher_model,
        save_path = "./results",
        batch_size = 16,
        device = torch.device(args.device),
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    )
    """
    configuration_student = config.Configuration(
        model_parameters=model_config,
        model = args.student_model,
        save_path = "./results",
        batch_size = 16,
        device = torch.device(args.device),
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.student_model, use_fast=True)
    )
    """
  
    teacher_model = SentenceTransformerWrapper.load_pretrained(args.teacher_model, params=configuration_teacher)
    #student_model = SentenceTransformerWrapper.load_pretrained(args.student_model, params=configuration_student)


    num_queries = args.num_queries

    corpus_percent = args.corpus_percentage

    document_dataset = config.load_file("../dataset/cached/jp-wikipedia-dataset")
    #document_dataset = DocumentCorpusDataset.from_tsv('../data/wiki-ja/ja.wikipedia_250k.txt')

    #TODO implement in dataloader
    all_sentences = list(document_dataset.sentences)
    random.shuffle(all_sentences)

    corpus_portion = all_sentences[:int(len(all_sentences)*corpus_percent)]
    print(f"Sentences present in the corpus: {len(corpus_portion)}")

    pipeline = APISearchPipeline(
        name="semantic_search", 
        embed_size=768,
        index_path="./index",
        corpus=corpus_portion, 
        model=teacher_model)

    while True:
        query = str(input("Enter a sentence to search"))
        results = pipeline(query, max_num_results=10)
        print(f"Results for query: {query}: ")
        for i, result in enumerate(results):
            print(f"result {i+1}: {result}")

    """
    queries = []

    for _ in range(args.num_queries):
        q = corpus_portion[random.randint(0, len(corpus_portion))]
        queries.append(q)

    print("#### Current Queries: ####")
    for q in queries:
        print(f"query: {q}")
    print()
    
    print("Extracting most relevant sentences with teacher model")
    pipeline_teacher = SemanticSearchPipeline(
        name="semantic_search", 
        index_path="./index",
        corpus=corpus_portion, 
        model=teacher_model)

    results_teacher = pipeline_teacher(queries, max_num_results=args.topk)
    print("Done.")
   
    print()
    print("Extracting most relevant sentences with student model")
    pipeline_student = SentenceMiningPipeline(name="paraphrase_mining", corpus=corpus_portion, model=student_model, corpus_chunk_size=len(corpus_portion))
    results_student = pipeline_student(queries, max_num_results=args.topk)
    print("Done.")
    print()
    print("Comparing student perfomance to the teacher...")
    compare_models(queries, results_teacher, results_student)
    print("Done.")
  

    print("Results: ")
    for k in results_teacher:
        print(f"Best matches for query: {queries[k]}")
        print(results_teacher[k])
        print()

    """





 



        