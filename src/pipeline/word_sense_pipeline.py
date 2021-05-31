"""
This script contains a pipeline to build sense embeddings with Transformer Encoder models
starting from a corpus of sentences, taking advantage of the information available in 
Lexical Knowledge Bases such as WordNet. 

WORKING ON A MIGRATION OF THE TEXT PROCESSING SECTION TO PYSPARK FOR QUITE THE SPEEDUP
(Although it would require writing custom Annotators, I guess)
"""


# STEP 1 - Data collection
#### Collect a corpus C of sentences

# STEP 2 - Contexts extraction
#### For a given lemma l pertaining to a synset s, find all occurences of l
#### in the corpus C and encode them with BERT 

# STEP 3 - Clustering
#### Cluster the contextualized vectors for any lemma l pertaining to a synset s 
#### using a clustering algorithm

# STEP 4 - Sense-cluster association (word-sense disambiguation)
#### Associate each cluster with one of the possible meanings (synset) of the lemma l and identify the cluster associated with the sense s
#### HOW? ARES uses UKB pageRank to do the job. A possible alternative could be to calculate a similarity score between the top sentences
#### of every cluster with the encoded gloss g of every synset s associated with the lemmas in the cluster with a cross-encoder model, 
#### effectively reducing the problem to a ranking problem

# STEP 5 - Contexts collection for every synset
#### Collect a set of contexts (sentences) for every lemma l for the sense s. 

# STEP 6 - Enriching contexts for every synset
#### Once we have a set of contexts for a synset s, we compute a vector representation for s 
#### HOW? Ares takes all the lemmas l' of every synset s' related to s in SyntagNet, and then, for every l and l', 
#### finds the sentences in corpus thant contain l and l' within a certain window (let's say window 3)
#### ALTERNATIVE

# STEP 7.1 - Sense embeddings (I)
#### We use SemCor corpus to compute, for every sense S of every synset syn, we compute the embedding representation
#### for S by averaging the BERT representations of every word in SemCor tagged with S

# STEP 7.2 - Sense embeddings (II)
#### We use WordNet glosses to build gloss embeddings. For every sense S of a synset syn, we build the gloss representation
#### by prepending all the lemmas of syn to the gloss g and encoding it with BERT
#### HOW . ARES computes the final embeddings this way: V[SC] || mean(V[G], V[S])
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

class WordSenseProcessingPipeline():
  """
  A pipeline to process the text in a format suitable
  to the building of sense embeddings
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



