"""
This script contains a pipeline to build sense embeddings with Transformer Encoder models
starting from a corpus of sentences, taking advantage of the information available in 
Lexical Knowledge Bases such as WordNet. 

WORKING ON BUILDING THE PIPELINE WITH PYSPARK FOR QUITE THE SPEEDUP
(Although it would require writing custom Annotators, I guess)
"""


# STEP 1 - Data collection
#### Collect a corpus C of sentences

# STEP 2 - Contexts extraction
#### For a given lemma l pertaining to a synset s, find all occurences of l
#### in the corpus C and encode them with BERT 
#### PROCEDURE: 
##### 1. For each sentence, map every token to its set of possible lemmas. Each lemma pertains to a certain synset, so we can
#####    always find the synset pertaining to the lemma from the lemma itself
##### 2. For each sentence, map every token in the sentence to a set of synsets. Each synset has a set of lemmas with the same sense
##### 3. For each token, compute its BERT embedding (subwords representations are merged via avg pooling)

# STEP 3 - Clustering and sense-cluster association
#### THE PROBLEM: We have to assign to every encoded word in a sentence a meaning (synsnet)
#### POSSIBLE SOLUTION: Cluster the contextualized vectors for any lemma l pertaining to a synset s 
#### using a clustering algorithm. This way, we can (carefully) affirm that the lemma l present
#### in the sentences of a cluster is used with the same meaning across the sentences, effectively
#### making the problem similar to a ranking problem.
#### PROCEDURE:
##### 1. For each cluster of any particular lemma l, compare the representation of the sentences part of the cluster with 
#####    the gloss of every possible sense of the lemma l. The highest scoring gloss will give us the most probable sense
#####    for the lemma l, and a sense tag for the sentences of the cluster

# STEP 4 - Contexts collection for every synset
#### Collect a set of contexts (sentences) for every lemma l for the sense s. 

# STEP 5 - Enriching contexts for every synset with related synsets
#### THE PROBLEM: Once we have a set of contexts for a synset s (that is, we have labeled our collection of lemmas with their 
#### most probavle senses), we compute a vector representation for s 
#### POSSIBLE SOLUTION: 
#### 1. Simply averaging all the representations of the lemmas tagged with the sense s, for every sense
####    found in the corpus
#### 2. Enriching the collection of lemmas for every sense s with other synsets related to s (that is, synsets that contain
####    lemmas that co-occur frequently with the lemmas of s) 

# STEP 6.1 - Sense embeddings (I)
#### We use SemCor corpus to compute, for every sense S of every synset syn, the embedding representation
#### for S by averaging the BERT representations of every word in SemCor tagged with S

# STEP 6.2 - Sense embeddings (II)
#### We use WordNet glosses to build gloss embeddings. For every sense S of a synset syn, we build the gloss representation
#### by prepending all the lemmas of syn to the gloss g and encoding it with BERT
#### HOW? ARES computes the final embeddings this way: V[SC] || mean(V[G], V[S])
#### where V[SC] are the SemCor embeddings, || is concatenation, V[G] are the gloss embeddings and V[S] are the synset embeddings

import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")
nltk.download("omw")
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional
import numpy as np
import torch
from src.utils.tokenizers import JapaneseTokenizer
from collections import defaultdict
from tqdm import tqdm
from src.modules.pyspark_extensions import (
    WordNetSynsetTransformer, 
    WordNetLemmaTransformer, 
    WordNetGlossTransformer
)


@dataclass
class WnSynset():
  name: str # the wn key
  lemmas: List[str]  
  gloss: str

  def __eq__(self, other):
    return self.name == other.name

@dataclass
class WnLemma():
  name: str #the lemma wn key
  synset: str #the synset its part of
  #embedding: Optional[Union[torch.Tensor, np.array]] = None

  def __eq__(self, other):
    return self.name == other.name

def get_all_wn_words() -> List[str]:
    return wn.words(lang="jpn")

class SparkWordSensePipeline():
  def __init__(self, annotators: list, stages: dict = {}):
    """
    params:
      :annotators list of pyspark annotators that make up a pipeline
      :stages collection of pipelines that need to be applied sequentially
    """
    self.annotators = annotators
    self.models = []
    self.stages = stages 

  @classmethod
  def from_annotators(cls, annotators):
    """
    Builds a PySpark pipeline to prepare the text for
    sense embeddings creation. Covers steps from 1 to 3
    """
    if not annotators:
      annotators = [
        DocumentAssembler() \
          .setInputCol("comment_text") \
          .setOutputCol("document")\
          .setCleanupMode('shrink'),

        Tokenizer() \
          .setInputCols(["document"]) \
          .setOutputCol("token"),

        StopWordsCleaner()\
          .setInputCols("normalized")\
          .setOutputCol("cleanTokens")\
          .setCaseSensitive(False),

        LemmatizerModel.pretrained() \
          .setInputCols(["cleanTokens"]) \
          .setOutputCol("lemma"),

        WordNetSynsetTransformer() \
          .setInputCols("lemma")\
          .setOutputCol("wn_synset"),

        WordNetLemmaTransformer() \
          .setInputCols("lemma")\
          .setOutputCol("wn_lemmas"),

        WordNetGlossTransformer() \
          .setInputCols("wn_synset")\
          .setOutputCol("wn_glosses"),

        BertEmbeddings.pretrained('bert_base_uncased', 'en') \
          .setInputCols("document", "token") \
          .setOutputCol("embeddings")\
      ]

    pipeline = Pipeline(stages=[*annotators])
    stages = {"default": pipeline}

    return cls(annotators, stages)

  def _check_stage_already_present(self, stage_id: str):
    if stage_id in self.stages:
      raise Exception("stage already present in the pipeline")

  def _check_state_not_present(self, stage_id: str):
    if stage_id not in self.stages:
      raise Exception("stage already present in the pipeline")
    
  def add_stage(self, stage_id: str, pipeline: Pipeline):
    self._check_stage_already_present(stage_id)
    self.stages[stage_id] = pipeline

  def fit(self, data, stage_id=None):
    if stage_id is None:
      stage_id = "default"
    self._check_state_not_present(stage_id)
    self.models.append(self.stages[stage_id].fit(data))

  def fit_all(self, data):
    for pipe in self.stages.values():
      self.models.append(pipe.fit(data))

  def transform(self, stage_id, data):
    self._check_state_not_present(stage_id)
    self.stages[stage_id].transform(data)
    

class WordSenseProcessingPipeline():
  """
  A pipeline to process the text by extracting lexical information
  from the WordNet graph and combine this information to build sense embeddings
  """
  def __init__(
    self, 
    corpus: List[str], 
    embedder: torch.nn.Module,
    tokenizer = JapaneseTokenizer()):
    self.corpus = corpus
    self.embedder = embedder
    self.tokenizer = tokenizer
    self.embeddings_map = {}
    print("Bulding sentences mapping...")
    self.sentences_map = {index : self.corpus[index] for index in range(len(self.corpus)) }
    print("Building tokens mapping...")
    self.tokens_map = self._build_tokens_map()
    print("Done.")
    print("Building lemmas mapping...")
    self.lemmas_map = self._build_lemmas_map()
    print("Done")

  def _build_tokens_map(self, processing_fn=None):
    mapping = defaultdict(list)
    for sent_index, sent in enumerate(self.corpus):
      if processing_fn is not None:
        sent = processing_fn(sent)
      mapping[sent_index].extend(self.tokenizer.tokenize(sent))
    return mapping

  def _build_lemmas_map(self):
    mapping = defaultdict(list)
    for sent_index, tokens in self.tokens_map.items():
      for w in tokens:
        for lemma in wn.lemmas(w):
          lemma_obj = WnLemma(
            name = lemma.name(),
            synset = lemma.synset(),
          )
          mapping[lemma_obj].append(sent_index)
    return mapping

  @property
  def synsets(self):
    added = set()
    for lemma in self.lemmas_map.keys():
      syn = lemma.synset()
      if syn not in added:
        added.add(syn)
        yield WnSynset(
            name = syn.name(),
            lemmas = [l.name() for l in syn.lemmas()],
            gloss = syn.definition()
        )

  @property
  def lemmas(self):
    for lemma in self.lemmas_map.keys():
      yield lemma



