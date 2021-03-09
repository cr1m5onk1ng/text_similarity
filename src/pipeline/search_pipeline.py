import os
from src.models.sentence_encoder import SentenceTransformerWrapper, SiameseSentenceEmbedder
from typing import Dict, List, Union
import torch
from torch import nn
import torch.nn.functional as F
import time
import hnswlib

class Pipeline:
    def __init__(self, name):
        self.name = name


class SearchPipeline(Pipeline):
    def __init__(self, corpus: Union[List[str], List[torch.Tensor], torch.Tensor], model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus = corpus 
        self.model = model

    def encode_corpus(self, documents: Union[List[str], torch.Tensor], return_embeddings: bool=False, convert_to_numpy=False) -> Union[List[str], torch.Tensor]:
        if isinstance(self.corpus, torch.Tensor):
            assert len(self.corpus.shape) == 2 #batch_size, embed_dim

        if isinstance(documents, list) and not return_embeddings:
            return self.model.encode_text(documents, output_np=convert_to_numpy)
        return documents

    def _search(self, queries: Union[List[str], torch.Tensor], max_num_results: int):
        raise NotImplementedError()

    def __call__(self, queries, max_num_results):
        return self._search(queries, max_num_results)


class SentenceMiningPipeline(SearchPipeline):
    def __init__(self, corpus_chunk_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus_chunk_size = corpus_chunk_size

    def search(
        self, 
        queries: Union[List[str], torch.Tensor], 
        max_num_results: int, 
        return_embeddings: bool = False) -> Dict[int, Union[List[str], torch.Tensor]]:

        print("#### Encoding queries ####")
        query_embeddings = self.encode_corpus(documents=queries, return_embeddings=return_embeddings)
        print("#### Queries encoded ####")

        print(f"Queries dimension: {query_embeddings.size()}")
        print()
        top_candidates = {}
        for corpus_index in range(0, len(self.corpus), self.corpus_chunk_size):
            corpus_chunk = self.corpus[corpus_index:self.corpus_chunk_size]
            if isinstance(self.corpus, list):
                print()
                print("##### Encoding corpus embeddings. This may take a while #####")
                start_time = time.time()
                if isinstance(self.model, SentenceTransformerWrapper):
                    corpus_chunk = self.model.encode_text(corpus_chunk)
                else:
                    corpus_chunk = self.model.encode(corpus_chunk, batch_size=16, convert_to_numpy=False, convert_to_tensor=True)
                end_time = time.time()
                print("#### Corpus encoded in {:.3f} seconds! ####".format(end_time - start_time))
                print()
            for query_idx, query_embedding in enumerate(query_embeddings):
                # expand to corpus dimension to calculate similarity scores
                # with all the corpus sentences
                query_embedding = query_embedding.unsqueeze(0).expand_as(corpus_chunk)
                scores = F.cosine_similarity(query_embedding, corpus_chunk, dim=-1)
                top_scores = torch.topk(scores, min(max_num_results, len(queries)), sorted=False)
                actual_candidates = top_scores[1]
                print(f"Top candidates indexes: {actual_candidates}")
                if return_embeddings:
                    assert isinstance(self.corpus, torch.Tensor)
                    top_candidates[query_idx] = self.corpus[torch.LongTensor(actual_candidates)]
                else:
                    candidates = []
                    for cidx in actual_candidates:
                        candidates.append(self.corpus[cidx])
                    top_candidates[query_idx] = candidates
        return top_candidates


    def __call__(self, queries: Union[List[str], torch.Tensor], max_num_results: int, return_embeddings: bool=False):
        return self._search(queries, max_num_results, return_embeddings)


class SemanticSearchPipeline(SearchPipeline):
    def __init__(
        self, 
        *args, 
        index_path: str = None, 
        ef: int = 50, 
        ef_construction: int = 400, 
        M: int = 64, 
        **kwargs):

        super().__init__(*args, **kwargs)
        self.index_path = index_path
        self.ef = ef
        self.ef_construction = ef_construction
        self.M = M

    def _search(
        self, 
        queries: Union[List[str], 
        torch.Tensor], 
        max_num_results: int, 
        return_embeddings: bool=False) -> Dict[int, List[str]]:
        
        #### INDEX PREPARATION ####
        print("####   Computing Query Embeddings...   ####")
        query_embeddings = self.encode_corpus(documents=queries, return_embeddings=return_embeddings, convert_to_numpy=True)
        print("Done.")
        print()

        embed_size = self.model.embedding_size
  
        index = hnswlib.Index(space = 'cosine', dim = embed_size)

        if os.path.exists(self.index_path):
            index.load_index(self.index_path)
        else:
            os.makedirs(self.index_path)
            print("#### Computing Corpus Embeddings. This may take a while. ####")
            corpus_embeddings = self.model.encode_text(self.corpus, output_np=True)
            print("Done.")
            print("Building the index. This may take a while...")
            index.init_index(
                max_elements = len(self.corpus), 
                ef_construction = self.ef_construction, 
                M = self.M)

            index.add_items(corpus_embeddings, list(range(corpus_embeddings.shape[0])))

            index.save_index(self.index_path)
            print("Done.")
        assert max_num_results < self.ef
        index.set_ef(self.ef)

        #### ACTUAL SEARCH ####
        top_results = {}

        for qidx, query in enumerate(query_embeddings):
            hits_ids, distances = index.knn_query(query, k=max_num_results)
            results = [(id, 1-score) for id, score in zip(hits_ids[0], distances[0])]
            results = sorted(results, key = lambda x: x[1], reverse=True)
            candidate_texts = []
            for hit_idx, _ in results:
                candidate_texts.append(self.corpus[hit_idx])
            top_results[qidx] = candidate_texts
        return top_results

    def __call__(
        self,
        queries: Union[List[str], torch.Tensor], 
        max_num_results: int, 
        return_embeddings: bool=False):

        return self._search(queries, max_num_results, return_embeddings)


class APISearchPipeline(SemanticSearchPipeline):
    def __init__(self, embed_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load index
        self.index = hnswlib.Index(space = 'cosine', dim = embed_size)
        index_load_path = os.path.join(self.index_path, "index.bin")
        if not os.path.exists(index_load_path):
            os.makedirs(self.index_path, exist_ok=True)
            print("No index found. Building index from corpus...")
            corpus_embeddings = self.model.encode_text(self.corpus, output_np=True)
            self.index.init_index(
                max_elements = len(self.corpus), 
                ef_construction = self.ef_construction, 
                M = self.M)
            self.index.add_items(corpus_embeddings, list(range(corpus_embeddings.shape[0])))
            index_save_path = os.path.join(self.index_path, "index.bin")
            self.index.save_index(index_save_path)
            print("Done.")
        else:
            assert os.path.exists(self.index_path)
            self.index.load_index(self.index_path)

    def _search(self, text: Union[str, List[str]], max_num_results: int, return_embeddings=False):
        if isinstance(text, str):
            text = [text]
        # encode query(ies)
        query_embeddings = self.model.encode_text(text, output_np=True)
        assert max_num_results < self.ef
        self.index.set_ef(self.ef)
        hits_ids, distances = self.index.knn_query(query_embeddings, k=max_num_results)
        # search in index
        results = [(id, 1-score) for id, score in zip(hits_ids[0], distances[0])]
        results = sorted(results, key = lambda x: x[1], reverse=True)
        candidate_texts = []
        for hit_idx, _ in results:
            candidate_texts.append(self.corpus[hit_idx])
        # return results
        assert candidate_texts
        return candidate_texts

