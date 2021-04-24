from typing import List, Tuple
from src.pipeline.search_pipeline import Pipeline
from src.models.sentence_encoder import SentenceTransformerWrapper
from src.utils import utils
from src.configurations import config
from src.dataset.dataset import DocumentCorpusDataset
from src.dataset.livedoor_dataset import LivedoorDataset
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import torch
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import gc
from googletrans import Translator
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from itertools import combinations
import random


def print_top_words_per_topic(top_words, top_topics):
        for topic_num in top_words:
            if topic_num != -1 and topic_num in top_topics:
                print(f"Most common words for topic: {topic_num}")
                word_and_score_list = top_words[topic_num]
                for w, _ in word_and_score_list:
                        print(f"\t{w}")   
                print()


class TopicModelingParams:
    def __init__(
        self, 
        n_neighbors = 15, 
        n_components = 5, 
        reduction_metric = "cosine", 
        min_cluster_size = 15, 
        cluster_metric = "euclidean"):

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reduction_metric = reduction_metric
        self.min_cluster_size = min_cluster_size
        self.cluster_metric = cluster_metric
        

#TODO Use wordNet hypernyms to find topics
class TopicModelingPipeline(Pipeline):
    def __init__(self, params, embedder: torch.nn.Module, documents: List[str], *args, lang = "jpn", **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        self.embedder = embedder
        self.documents = documents
        self.lang = lang
        self.translator = Translator()

    def __call__(self, max_n_words=10, reduce_topics=None):
        print("Embedding sentences. this may take a while...")
        embeddings = self.embedder.encode_text(self.documents, output_np=True)
        print("Done")
        print()
        print(f"Reducing embeddings dimension to {self.params.n_components}. this may take a while...")
        reduced_embeddings = self._reduce_dim(embeddings)
        print("Done.")
        print()
        print("Clustering embeddings. This may take a while")
        cluster = self._cluster(reduced_embeddings)
        print("Done.")
        tf_idf, count = self._c_tf_idf()
        docs_df = self._create_docs_df(cluster)
        if reduce_topics is not None:
            return self.reduce_n_topics(tf_idf, docs_df, n=reduce_topics)
        docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
        top_n_words = self.extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=max_n_words)
        topic_sizes = self.extract_topic_sizes(docs_df)
        return top_n_words, topic_sizes

    def _create_docs_df(self, cluster):
        docs_df = DataFrame(self.documents, columns=["Doc"])
        docs_df['Topic'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        return docs_df

    def _reduce_dim(self, embeddings: np.array, save=True) -> np.array:
        umap_embeddings = umap.UMAP(n_neighbors=self.params.n_neighbors, 
                            n_components=self.params.n_components, 
                            metric=self.params.reduction_metric).fit_transform(embeddings)
        if save:
            utils.save_file(umap_embeddings, "../embeddings/reduced", "umap_embeddings.bin")
        return umap_embeddings

    def _cluster(self, reduced_embeddings: np.array, save=True):
        cluster = hdbscan.HDBSCAN(min_cluster_size=self.params.min_cluster_size,
                          metric=self.params.cluster_metric,                      
                          cluster_selection_method='eom').fit(reduced_embeddings)
        if save:
            utils.save_file(cluster, "../embeddings/reduced", "hdbscan_cluster.bin")
        return cluster

    def _c_tf_idf(self, ngram_range=(1, 1)) -> Tuple[np.array, CountVectorizer]:
        m = len(self.documents)
        count = CountVectorizer(
            ngram_range=ngram_range, 
            stop_words="english" if self.lang == "eng" else None
            ).fit(self.documents)
        t = count.transform(self.documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        return tf_idf, count

    def _find_wn_least_common_hypernyms(self, synset_pairs: List[Tuple], return_depth=False):
        ancestors = set()
        ancestors_dict = {}
        for syn1, syn2 in synset_pairs:
            if return_depth:
                ancestors_dict.update(syn1._shortest_hypernym_paths(syn2))
            ancestors.update(syn1.lowest_common_hypernyms(syn2))
        return ancestors if not return_depth else ancestors_dict

    def reduce_n_topics(self, tf_idf: np.array, docs_df, n: int=10):
        for i in range(n):
            # Calculate cosine similarity
            similarities = cosine_similarity(tf_idf.T)
            np.fill_diagonal(similarities, 0)

            # Extract label to merge into and from where
            topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
            topic_to_merge = topic_sizes.iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Adjust topics
            docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            old_topics = docs_df.sort_values("Topic").Topic.unique()
            map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
            docs_df.Topic = docs_df.Topic.map(map_topics)
            docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
            # Calculate new topic words
            m = len(self.documents)
            tf_idf, count = self._c_tf_idf(docs_per_topic.Doc.values, m)
            top_n_words = self._extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=n)
        topic_sizes = self.extract_topic_sizes(docs_df)
        return top_n_words, topic_sizes

    def extract_topic_sizes(self, df) -> DataFrame:
        topic_sizes = (df.groupby(['Topic'])
                         .Doc
                         .count()
                         .reset_index()
                         .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                         .sort_values("Size", ascending=False))
        return topic_sizes

    def extract_top_n_words_per_topic(self, tf_idf: np.array, count_vec: CountVectorizer, docs_per_topic: int, n=10):
        words = count_vec.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        gc.collect()
        return top_n_words

    def find_general_categories(self, top_words_dict, top_topics):
        for topic in top_words_dict:
            if topic in top_topics:
                synsets = []
                word_and_score_list = top_words_dict[topic]
                for w, _ in word_and_score_list:
                    syns = wn.synsets(w, lang=self.lang, pos="n")
                    synsets += syns
                synsets_pairs = combinations(synsets, 2)
                common_hypernyms = self._find_wn_least_common_hypernyms(synsets_pairs)
                for hyp in common_hypernyms:
                    print(f"\t{hyp}")
                """
                hyp_pairs = combinations(common_hypernyms, 2)
                hyps_with_depths = self._find_wn_least_common_hypernyms(hyp_pairs, return_depth=True)
                print(f"Topic {topic} most common hypernyms:")
                
                for hyp in hyps_with_depths:
                    print(f"\t{hyp} with depth: {hyps_with_depths[hyp]}")
                """
                    
    def print_top_words_per_topic(self, top_words, top_topics, translate=False):
        for topic_num in top_words:
            if topic_num in top_topics:
                print(f"Most common words for topic: {topic_num}")
                word_and_score_list = top_words[topic_num]
                print(word_and_score_list)
                for w, _ in word_and_score_list:
                    if translate:
                        try:
                            translated = self.translator.translate(w)
                            print(f"\t{translated.text}")
                        except:
                            continue
                    else:
                        print(f"\t{w}")   
                print()
    
    @staticmethod
    def plot_clusters(embeddings: np.array, cluster):
        umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        result = DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = cluster.labels_
        
        # Visualize clusters
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()
 


if __name__ == "__main__":

    model = "sentence-transformers/quora-distilbert-multilingual"

    """

    model_config = config.ModelParameters(
        model_name = "eval_sentence_mining"
    )

    model_config = config.Configuration(
        model_parameters=model_config,
        model = model,
        save_path = "./results",
        batch_size = 16,
        sequence_max_len = 512,
        device = torch.device("cuda"),
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=True)
    )

    encoder = SentenceTransformerWrapper.load_pretrained(model, params=model_config)
    
    #document_dataset = DocumentCorpusDataset.from_tsv("../data/wiki-ja/ja.wikipedia_250k.txt")
    #document_dataset = LivedoorDataset.from_collection("../data/livedoor/text/").paragraphs

    perc = 0.25

    #articles = document_dataset[0:int(len(document_dataset) * perc)]
    #articles = document_dataset.articles[0:int(len(document_dataset) * perc)]
    dataset = fetch_20newsgroups(subset='all')['data']
    random.shuffle(dataset)
    articles = dataset[0:int(len(dataset) * perc)]

    params = TopicModelingParams(min_cluster_size = 15)

    pipeline = TopicModelingPipeline(
        name = "topic_modeling",
        params = params,
        embedder = encoder,
        documents = articles,
        lang = "eng"
    )

    top_n_words, topic_sizes = pipeline(max_n_words = 20, reduce_topics=None)
    top_topics = [t for t in topic_sizes.head(10).Topic if t!= -1]

    print(f"Top topics: {top_topics}")

    print(f"Topic Sizes: {topic_sizes}")
    pipeline.print_top_words_per_topic(top_n_words, top_topics, translate = False)
    print("\n\n")
    print(f"Common Hypernyms")
    pipeline.find_general_categories(top_n_words, top_topics)
    """
    dataset = fetch_20newsgroups(subset='all')['data']
    topic_model = BERTopic(embedding_model=model, nr_topics=10)
    topics, _ = topic_model.fit_transform(dataset)
    top_topics = topic_model.get_topic_freq().head(10).Topic
    print_top_words_per_topic(topic_model.get_topics(), top_topics)

    #utils.save_file(top_n_words, "./results", "japanese_wiki_topics.bin")