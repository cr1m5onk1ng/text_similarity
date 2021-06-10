"""
This file contains an experiment aiming to construct Word-level sense-aware embeddings
with the help of multimodal models like CLIP, contextualized embedders like BERT, and lexical
sense inventories like WordNet.

Papers:
GlossBert: https://arxiv.org/abs/1908.07245
EVilBert: https://www.ijcai.org/proceedings/2020/67
LMMS: https://www.aclweb.org/anthology/P19-1569/
CLIP: https://arxiv.org/abs/2103.00020
"""

####### STEPS ########

# 1. Dataset Building
##   For every word w in WordNet, we want to build a dataset that maps w
##   to a set of images and sentences related to w
##   - For every word w in WordNet, search for wikipedia pages related to the word and scrape
##     a maximum N number of images, their captions, and a maximun M number of sentences 
##     in the article containing the word or any of its lemmas
##   - Finding related pages: Several strategies are possible:
##     1. Use a massive dataset such as WIT
##        * Find all the articles in the dataset that contain the word w in their title
##        * Use the Wikimedia API to go fetch the article content to scrape the sentences
##        * Don't need to scrape the images, already present in the datasets
##        * Profit
##     2. Build from scratch
##        * Use Wikipedia API to fetch all the articles that contain the word w in the title
##        * For any article d extracted for the word w, compare every gloss g for every synset s
##          of w and the description t of the article d with a cross-encoder. For every g, associate
##          to g the article with the highest score (thus obtaining a mapping from s to d)
##

# 2. Sense Tagging
##   For every lemma l, we want to associate to l its corresponding synset s.
##   In order to do this, we take advantage of the representational power of CLIP.
##   - For every word w in WordNet, we retrieve the set of its possible WordNet synsets S. 
##   - We retrieve the set of all sentences containing w, that we call T,
##     and the set of all the images associated to w, that we call I
##   - For every w, we construct a set of prompts in the following way: 
##     We retrieve all the possible synsets for w. For each image i, we
##     take the caption of i, let's call it c, and construct a set of prompts
##     C by appending the gloss of each possible synset s for w to the caption.
##     The gloss is put within parentheses as a weak signal to the model.
##     Alternatively, we can completely discard the caption and only use
##     the gloss as text label (prepended with "a photo of" and the main synset lemma). 
##     In case no caption is available for i, we poll a certain number of sentences associated 
##     with w and related to i and take their avarage representation.
##   - Now that we have a set of prompts for w and a set of images, we embed every
##     prompt and image with CLIP
##   - We calculate the similary between every prompt-image pair and take the 
##     highest scoring one. The gloss of the highest scoring prompt will tell us
##     what is the best synset candidate for the word w and image i, thus obtaining 
##     a sense tag for w and i.

# 3. Contrastive Word Representation Learning
##   Now that we have sense tags for the words in our corpus, we want to train a word
##   encoder that takes into account word sense and the general concepts related to that word.
##   In order to achieve this, we leverage all the information we have gathered in the previous steps.
##   

"""
    We want to build a dataset such that, for every (non-rare) word
    in WordNet, we have:
        1. A set of contexts in which the word appears IN ANY OF ITS POSSIBLE SENSES
        2. A set of images related to those possible senses

    How we do it:

    STEP I - Articles Ranking based on gloss information
        1. For every synset s in WordNet, we take its gloss g
        2. We want to extract all the articles related to a word. To do that, we start
           from a word w and get his WordNet synsets. For every synset s, we take all the
           articles that contain the main lemma of s (with some filtering based on NER tagging) 
           in the title and rank them with respect to g. 
           The ranking process works like this:
            - we use the gloss g as query for the first paragraph of every article
              extracted for a lemma l.
            - we retrieve N number of most similar sentences with a Bi-Encoder
            - we calculate a ranking score between every sentence and g using a Cross-Encoder
            - we take the avarage of that score
            - we repeat the process for every cluster of l
            - once we have a ranking score for every cluster/gloss pair, we associate
              each gloss (and thus each synset) to the highest scoring cluster
        3. We take the most N most relevant articles for every g and associate them to s

    STEP II - Context extraction
        1. For every synset s for a word w, we want to find all the sentences in the associated articles
           that contain the main lemma of s, thus obtaining a corpus C for s
        2. For every corpus associated to a synset s, we encode the sentences with BERT,
            thus obtaining a word embedding for every instance of the main lemma l for the
            synset s

    STEP III - Clustering
        1. Now that we have a set of embedded contexts for the main lemma of
           every synset s, we have to take another step to ensure the lemma
           contained in the sentences refers to the sense expressed by s. In fact,
           even if we consider the ranking step to give us a good indication of the sense, it
           may happen (and it surely happens in practice) that, even though
           an article is related to a synset, not all the instances of the main lemma l
           in the article refer to the sense expressed by the synset.
           In order to add another disambiguation step, we cluster the representations
           for the main lemma l with a k-means clustering algorithm.
        2. For each cluster of the lemma l, we take N sentences from the cluster and compare
           them to every gloss g of every possible sense s of l with a cross-encoder. The highest
           scoring gloss will give us a sense tag for the cluster, thus associating the sentences
           in the cluster with the synset s

    STEP IV - Multimodal Sense Embeddings

"""


from typing import List, Union
from .word_sense_pipeline import SparkWordSensePipeline
from .ranking_pipeline import RankingPipeline
from .clustering import ClusteringPipeline
import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")
nltk.download("omw")
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, BooleanType
from pyspark.sql.functions import udf


class SparkWordSenseMultimodalPipeline(SparkWordSensePipeline):
    """
    Preprocessing steps:
        1. Collect words of interest (default all WordNet non-rare words)
        2. Map words to possible synsets
        3. Map synsets to glosses
        4. Map words to articles
        5. Filter articles titles 
            Pick articles that:
            - contain the word exactly once
            - contain the word as verb or noun
            - contain the word, which doesn't refer to some person or institution
    """
    def __init__(self, data, params, bi_encoder, cross_encoder, *args, words=None, filters={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.ranker = RankingPipeline(
            params = params,
            model = bi_encoder,
            cross_encoder = cross_encoder
        )
        self.clusterer = ClusteringPipeline(
            method = "k-means",
            n_clusters = params.n_clusters
        )
        self.filters = filters
        # if no words given, default is all wordnet words
        if words is None:
            self.words = set(list(wn.words()))
        else:
            self.words = set(words)

    def _title_contains_lemma(self, title_tokens: List[str]):
        for t in title_tokens:
            if t in self.words:
                return True
        return False

    def _find_lemma_for_article(self, title_tokens):
        for t in title_tokens:
            if t in self.words:
                return t 

    def _cache_filter(self, filter_name, filter):
        self.filters[filter_name] = filter

    def _filter_by_lemmas(self):
        """
        we wanna leave out all the articles that don't have the lemmas
        we are looking for in the title
        """
        if "contains_lemma" in self.filters:
            filter_by_lemma = self.filters["contains_lemma"]
        else:
           filter_by_lemma = udf(self._title_contains_lemma, BooleanType())
           self._cache_filter("contains_lemma", filter_by_lemma)
        self.data = self.data.filter(filter_by_lemma('title'))

    def _filter_by_pos(self):
        raise NotImplementedError()

    def _filter_by_ner(self):
        raise NotImplementedError()

    def _map_article_to_lemma(self):
        if "map_title_to_lemma" in self.filters:
            map_title_to_lemma = self.filters["map_title_to_lemma"]
        else:
            map_title_to_lemma = udf(self._find_lemma_for_article, StringType()) 
            self._cache_filter("map_title_to_lemma", map_title_to_lemma)
        self.data = self.data.withColumn("title_lemma", udf('title'))

    def add_filter(self, name, function, return_type):
        self.filters[name] = udf(function, return_type)





