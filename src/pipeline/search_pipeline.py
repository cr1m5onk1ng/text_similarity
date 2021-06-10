import os
from src.configurations.config import SearchConfiguration
from src.models.sentence_encoder import SentenceTransformerWrapper
from typing import Dict, List, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
import time
import hnswlib
import onnxruntime
import numpy as np
from tqdm.std import trange

class Pipeline:
    def __init__(self, params: SearchConfiguration, model: nn.Module):
        self.params = params
        self.model = model

    def encode_corpus(self, documents: Union[List[str], torch.Tensor], convert_to_numpy=False) -> Union[List[str], torch.Tensor]:
        if isinstance(documents, list):
            return self.model.encode_text(documents, output_np=convert_to_numpy)
        return documents

class SearchPipeline(Pipeline):
    def __init__(self, *args, corpus: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus = corpus

    def _index(self, corpus: List[str]):
        raise NotImplementedError()

    def _search(self, queries: Union[List[str], torch.Tensor], max_num_results: int):
        raise NotImplementedError()

    def __call__(self, queries, max_num_results):
        return self._search(queries, max_num_results)


class SentenceMiningPipeline(SearchPipeline):
    def __init__(self, corpus_chunk_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus_chunk_size = corpus_chunk_size

    def _search(
        self, 
        queries: Union[List[str], torch.Tensor], 
        corpus: List[str],
        max_num_results: int, 
        return_embeddings: bool = False) -> Dict[int, Union[List[str], torch.Tensor]]:

        print("#### Encoding queries ####")
        query_embeddings = self.encode_corpus(documents=queries, return_embeddings=return_embeddings)
        print("#### Queries encoded ####")

        print(f"Queries dimension: {query_embeddings.size()}")
        print()
        top_candidates = {}
        if corpus is not None:
            self.corpus = corpus
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
                top_scores = torch.topk(scores, min(max_num_results, len(queries)), dim=1, sorted=False, largest=True)
                actual_candidates = top_scores[1]
                print(f"Top candidates indexes: {actual_candidates}")
                if return_embeddings:
                    assert isinstance(self.corpus, torch.Tensor)
                    top_candidates[query_idx] = self.corpus[torch.LongTensor(actual_candidates)]
                else:
                    candidates = []
                    for cidx in actual_candidates:
                        candidates.append((cidx, self.corpus[cidx]))
                    top_candidates[query_idx] = candidates
        return top_candidates


    def __call__(self, queries: Union[List[str], torch.Tensor], max_num_results: int):
        return super().__call__(queries, max_num_results)


class SemanticSearchPipeline(SearchPipeline):
    def __init__(
        self, 
        index_path,
        *args, 
        **kwargs):

        super().__init__(*args, **kwargs)
        self.index_path = index_path
        self.index = hnswlib.Index(space = 'cosine', dim = self.params.model_parameters.hidden_size)
        if os.path.exists(os.path.join(self.index_path, "index.bin")):
            self.index.load_index(self.index_path)
        else:
            self._index(self.corpus)

    def _index(self, corpus: List[str]):
        os.makedirs(self.index_path)
        print("#### Computing Corpus Embeddings. This may take a while. ####")
        corpus_embeddings = self.encode_corpus(self.corpus, convert_to_numpy=True)
        print("Done.")
        print("Building the index. This may take a while...")
        self.index.init_index(
            max_elements = len(corpus), 
            ef_construction = self.params.ef_construction, 
            M = self.params.M)
        self.index.add_items(corpus_embeddings, list(range(corpus_embeddings.shape[0])))
        self.index.save_index(self.index_path)
        print("Done.")
        self.index.set_ef(self.params.ef)

    def _search(
        self, 
        queries: Union[List[str], torch.Tensor], 
        max_num_results: int) -> Dict[int, List[str]]:

        assert max_num_results < self.params.ef, "ef hyperparam should less than the maximun number of results"

        query_embeddings = self.encode_corpus(queries, convert_to_numpy=True)

        top_results = {}

        for qidx, query in enumerate(query_embeddings):
            hits_ids, distances = self.index.knn_query(query, k=max_num_results)
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
        max_num_results: int):

        return self._search(queries, max_num_results)

    def add_to_index(self, text: Union[str, List[str]]):
        embeddings = self.encode_corpus(text, convert_to_numpy=True)
        added_length = embeddings.shape[0]
        self.index.resize_index(self.params.num_elements + added_length)
        labels = np.arange(self.params.num_elements, self.params.num_elements + added_length)
        self.index.add_items(embeddings, labels, num_threads = -1)
        self.params.num_elements += added_length

    def remove_from_index(self, ids):
        for id in ids:
            try:
                self.index.mark_deleted(id)
                self.params.num_removed += 1
            except RuntimeError:
                # can't find id, continue
                continue

    def num_indexed(self):
        """
        returns the current number of indexed embeddings
        """
        return self.index.get_current_count() - self.params.num_removed


class APISearchPipeline(SemanticSearchPipeline):
    def __init__(
        self, 
        params,
        max_n_results: int, 
        *args, 
        inference_mode: bool=True, 
        session_options: bool=None,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        self.inference_mode = inference_mode
        self.sess_options = session_options
        self.max_n_results = max_n_results
        if self.sess_options is None:
            self.sess_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.params.model_path, self.sess_options)

    def __call__(
        self,
        queries: Union[List[str], torch.Tensor], 
        max_num_results: int):
        return self._search(queries, max_num_results)
       
    def encode_corpus(self, documents: List[str]) -> Union[torch.Tensor, np.array]:
        length_sorted_idx = np.argsort([len(sen) for sen in documents])
        documents = [documents[idx] for idx in length_sorted_idx]
        encoded_documents = []
        for start_index in trange(0, len(documents), self.params.batch_size):
            sentences_batch = documents[start_index:start_index+self.params.batch_size]   
            encoded_dict = self.params.tokenizer(
                    text=sentences_batch,
                    add_special_tokens=True,
                    padding='longest',
                    truncation=True,
                    max_length=self.params.sequence_max_len,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors='np'
            )
            inputs = {
            'input_ids': encoded_dict["input_ids"].reshape(1, -1),
            'attention_mask': encoded_dict["attention_mask"].reshape(1, -1),
            }
            output = self.session.run(None, inputs)
            embeddings = output[0]
            encoded_documents.extend(embeddings)
        encoded_documents = [encoded_documents[idx] for idx in np.argsort(length_sorted_idx)]
        return encoded_documents
     

    

