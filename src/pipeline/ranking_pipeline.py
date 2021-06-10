from typing import List
from .search_pipeline import SentenceMiningPipeline

class RankingPipeline(SentenceMiningPipeline):
    """
    Ranks a set of documents with respect to a query text
    using a Bi-Encoder to search for the most similar documents
    and then a Cross-Encoder to re-rank the documents for added accuracy.
    """
    def __init__(self, cross_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_encoder = cross_encoder

    def _rank(self, queries: List[str], corpus: List[str], top_k=5, search_first=True) -> List[dict]:
        ##### Sematic Search #####
        #Encode the query using the bi-encoder and find potentially relevant passages
        results = []
        if search_first:
            query_embeddings = self.model.encode_text(queries, convert_to_tensor=True)
            query_embeddings = query_embeddings.cuda()
            hits = self._search(query_embeddings, corpus, max_num_results=top_k)
            hits = hits[0]  # Get the hits for the first query
            to_rank = [hit[1] for hit in hits]
        else:
            to_rank = corpus
        ##### Re-Ranking #####
        #Now, score all retrieved passages with the cross_encoder
        for query in queries:
            cross_inp = [[query, el for el in to_rank]]
            cross_scores = self.cross_encoder.predict(cross_inp)
            #Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            avg_score = sum(cross_scores) / len(cross_scores)

            results.append( {
                'results': hits,
                'cross_scores': cross_scores,
                'avg_score': avg_score
            } )
            
        return results

    def __call__(self, queries: List[str], corpus: List[str], top_k=5, search_first=True):
        return self._rank(queries, corpus, top_k, search_first)
