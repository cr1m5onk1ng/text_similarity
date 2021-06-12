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
##     a maximum N number of images, the article description (first paragraph of first section) 
##     and a maximun M number of sentences in the article containing the word or any of its lemmas
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
        2. A set of images related to the possible senses

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
from .word_sense_pipeline import SparkPipelineWrapper
from ..ranking_pipeline import RankingPipeline
from ..clustering import ClusteringPipeline
from src.modules.pyspark_extensions import FilterArticlesByLemmaTransformer, MapTitleToLemmaTransformer
import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")
nltk.download("omw")
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, BooleanType
from pyspark.sql.functions import udf
from sparknlp.annotator import WordEmbeddingsModel, PerceptronModel, NerDLModel, TextMatcher
import os


class SparkWordSenseMultimodalPipeline(SparkPipelineWrapper):
    """
    Preprocessing steps:
        1. Collect words of interest (default all WordNet non-rare words)
        2. Map words to possible synsets
        3. Map synsets to glosses
        4. Filter articles titles 
            Pick articles that:
            - contain the word exactly once
            - contain the word as verb or noun
            - contain the word, which doesn't refer to some specific entity 
    At the end of the preprocessing step, we want to obtain a dataframe containing:
        1. Articles titles (+url)
        2. Articles description 
        3. WordNet lemma of the title
        4. POS of the lemma
        5. NER of the lemma
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

    def _cache_filter(self, filter_name, filter):
        self.filters[filter_name] = filter

    def _write_lemmas(self, filepath, name="lemmas.txt"):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(os.path.join(filepath, name), 'w') as f:
            for lemma in self.words:
                f.write(lemma + '\n')

    def _add_lemma_filter(self, input_col, use_matcher=False, lemmas_path=None):
        """
        we wanna leave out all the articles that don't have the lemmas
        we are looking for in the title
        """
        if use_matcher:
            if lemmas_path is None:
                raise Exception("No file specified for the lemmas")
            if not os.path.exists(lemmas_path):
                self._write_lemmas(lemmas_path)
            annotator = TextMatcher() \
                .setInputCols(["document",'token']) \
                .setOutputCol("matched_text") \
                .setCaseSensitive(False) \
                .setEntities(path=os.path.join(lemmas_path, "lemmas.txt"))
        else:
            annotator = FilterArticlesByLemmaTransformer \
                .setInputCol(input_col) \
                .setLemmas(self.words)

        self.add_annotator(annotator)

    def _add_pos(self, pos_model, input_cols, output_col, lang="en"):
        self.add_annotators(
            [
                PerceptronModel.pretrained(pos_model, lang) \
                    .setInputCols(input_cols) \
                    .setOutputCol(output_col)
            ]
        )

    def _add_embeddings(self, embed_model, input_cols, output_col):
        self.add_annotators(
            [
                WordEmbeddingsModel.pretrained(embed_model) \
                    .setInputCols(input_cols) \
                    .setOutputCol(output_col)
            ]
        )

    def _add_ner(self, ner_model, input_cols, output_col):
        self.add_annotators(
            [
                NerDLModel.pretrained(ner_model) \
                    .setInputCols(input_cols) \
                    .setOutputCol(output_col)
            ]
        )

    def _map_article_to_lemma(self, input_col, output_col):
        self.add_annotators(
            [
                MapTitleToLemmaTransformer() \
                    .setInputCol(input_col) \
                    .setOutputCol(output_col) \
                    .setLemmas(self.words)
            ]
        )
    
    def add_filter(self, name, function, return_type):
        self.filters[name] = udf(function, return_type)

    def filter_by_pos(self):
        self.data = self.data.withColumn('cols', 
                  F.explode(
                      F.arrays_zip(
                          'pos.result',
                          'pos.begin',
                          'token.begin',
                      ) 
                  )
        ) \
        .withColumn("pos_result", F.expr("cols['0']")) \
        .withColumn("pos_begin", F.expr("cols['1']")) \
        .withColumn("token_begin", F.expr("cols['2']")) \
        .filter((F.col('pos_result') == 'NN') & (F.col('pos_begin') == F.col('token_begin'))) \
        .drop("cols", "pos_result", "pos_begin", "token_begin")

    def filter_by_ner(self, column):
        self.data = self.data.withColumn('cols', 
                  F.explode(
                      F.arrays_zip(
                          'ner.result',
                          'ner.begin',
                          'token.begin',
                      ) 
                  )
        ) \
        .withColumn("ner_result", F.expr("cols['0']")) \
        .withColumn("ner_begin", F.expr("cols['1']")) \
        .withColumn("token_begin", F.expr("cols['2']")) \
        .filter((F.col('ner_result') == 'O') & (F.col('ner_begin') == F.col('token_begin'))) \
        .drop("cols", "ner_result", "ner_begin", "token_begin")

    





