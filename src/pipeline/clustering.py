from typing import List, Union
from sklearn.cluster import KMeans
from .search_pipeline import Pipeline
import torch
import numpy as np
from collections import defaultdict

class ClusteringPipeline(Pipeline):
    def __init__(self, n_clusters, *args, method="k-means", **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.n_clusters = n_clusters
        if method == "k-means":
            self.clusterer = KMeans(n_clusters=self.n_clusters)
    
    def _cluster(self, corpus: Union[List[str], torch.Tensor, np.array]):
        if isinstance(corpus, list):
            corpus = self.encode_corpus(corpus)
        self.clusterer.fit(corpus)
        clusters = self.model.labels_
        results = defaultdict(list)
        for text_id, cluster_id in enumerate(clusters):
            results[cluster_id].append(corpus[text_id])

