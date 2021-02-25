import os
from src.models.sentence_encoder import SiameseSentenceEmbedder
from typing import Dict, List, Union
import torch
from torch import nn
import torch.nn.functional as F
import hnswlib

class Pipeline:
    def __init__(self, name):
        self.name = name


class SearchPipeline(Pipeline):
    def __init__(self, corpus: Union[List[str], List[torch.Tensor], torch.Tensor], model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus = corpus 
        self.model = model

    def encode_corpus(self, documents: Union[List[str], torch.Tensor], return_embeddings: bool=False) -> Union[List[str], torch.Tensor]:
        if isinstance(self.corpus, torch.Tensor):
            assert len(self.corpus.shape) == 2 #batch_size, embed_dim

        if isinstance(documents, list) and not return_embeddings:
            if isinstance(self.model, SiameseSentenceEmbedder):
                return self.model.encode_text(documents)
            else:
                return self.model.encode(documents, batch_size=16, convert_to_numpy=False, convert_to_tensor=True)
        return documents

    #TODO
    def load_corpus(self, path):
        pass


class SentenceMiningPipeline(SearchPipeline):
    def __init__(self, corpus_chunk_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus_chunk_size = corpus_chunk_size

    def _mine(
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
                print("##### Encoding corpus embeddings. This may take a while #####")
                if isinstance(self.model, SiameseSentenceEmbedder):
                    corpus_chunk = self.model.encode_text(corpus_chunk)
                else:
                    corpus_chunk = self.model.encode(corpus_chunk, batch_size=16, convert_to_numpy=False, convert_to_tensor=True)
                print("#### Corpus encoded! ####")
                print(f"Corpus shape: {corpus_chunk.shape}")
                print()
            for query_idx, query_embedding in enumerate(query_embeddings):
                # expand to corpus dimension to calculate similarity scores
                # with all the corpus sentences
                print(f"Query dimension: {query_embeddings.shape}")
                query_embedding = query_embedding.unsqueeze(0).expand_as(corpus_chunk)
                scores = F.cosine_similarity(query_embedding, corpus_chunk, dim=-1)
                print(f"Scores dimension: {scores.shape}")
                top_scores = torch.topk(scores, min(max_num_results, len(queries)), sorted=False)
                print(f"Top candidates indexes: {top_scores[1]}")
                if return_embeddings:
                    assert isinstance(self.corpus, torch.Tensor)
                    top_candidates[query_idx] = self.corpus[torch.LongTensor(top_scores[1])]
                else:
                    candidates = []
                    for cidx in top_scores[1]:
                        candidates.append(self.corpus[cidx])
                    top_candidates[query_idx] = candidates
        return top_candidates


    def __call__(self, queries: Union[List[str], torch.Tensor], max_num_results: int, return_embeddings: bool=False):
        return self._mine(queries, max_num_results, return_embeddings)


class SemanticSearchPipeline(SearchPipeline):
    def __init__(
        self, 
        index_path: str, 
        *args, 
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
        query_embeddings = self.encode_corpus(documents=queries, return_embeddings=return_embeddings)
        print("Done.")
        print()

        if isinstance(self.model, SiameseSentenceEmbedder):
            embed_size = self.model.context_embedder.config.hidden_size
        else:
            embed_size = self.model.config.hidden_size
        index = hnswlib.index(space = 'cosine', dim = embed_size)

        if os.path.exists(self.index_path):
            index.load_index(self.index_path)
        else:
            print("#### Computing Corpus Embeddings. This may take a while. ####")
            corpus_embeddings = self.model.encode_text(self.corpus)
            print("Done.")
            index.init_index(
                max_elements = len(self.corpus), 
                ef_construction = self.ef_construction, 
                M = self.M)

            index.add_items(corpus_embeddings, list(range(corpus_embeddings.shape[0])))

            index.save_index(self.index_path)
        assert max_num_results < self.ef
        index.set_ef(self.ef)

        #### ACTUAL SEARCH ####
        top_results = {}

        for qidx, query in enumerate(query_embeddings):
            hits_ids, distances = index.knn_query(query, k=max_num_results)
            results = [(id, 1-score) for id, score in zip(hits_ids, distances)]
            results = sorted(results, key = lambda x: x[1], reverse=True)
            candidate_texts = []
            for hit_idx, _ in results:
                candidate_texts.append(self.corpus[hit_idx])
            top_results[qidx] = candidate_texts

    def __call__(
        self,
        queries: Union[List[str], 
        torch.Tensor], 
        max_num_results: int, 
        return_embeddings: bool=False):

        return self._search(queries, max_num_results, return_embeddings)