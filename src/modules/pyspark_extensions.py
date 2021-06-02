import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")
nltk.download("omw")
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as F
from sparknlp.pretrained import BertEmbeddings


class WordNetLemmaTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable):
    """
    PySpark Transformer that maps words to a list of WordNet lemmas
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        module = __import__("__main__")
        setattr(module, 'WordNetLemmaTransformer', WordNetLemmaTransformer)
        super(WordNetLemmaTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def lemmas(self, word):
        for lemma in wn.lemmas(word):
            yield lemma.name()

    def _transform(self, dataset):
         # User defined function to map every word to a list of lemmas
        word2lemmas = udf(lambda word: list(self.lemmas(word)), ArrayType(StringType()))

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        #return dataset.withColumn(out_col, F.explode(text2lemmas(in_col)))
        return dataset.withColumn(out_col, word2lemmas(in_col))


class WordNetSynsetTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable):
    """
    PySpark Transformer that maps words to a list of WordNet synsets
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        module = __import__("__main__")
        setattr(module, 'WordNetSynsetTransformer', WordNetSynsetTransformer)
        super(WordNetSynsetTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def synsets(self, word):
        for synset in wn.synsets(word):
            yield synset.name()

    def _transform(self, dataset):
         # User defined function to map every word to a list of lemmas
        word2synsets = udf(lambda word: list(self.synsets(word)), ArrayType(StringType()))

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        #return dataset.withColumn(out_col, F.explode(text2lemmas(in_col)))
        return dataset.withColumn(out_col, word2synsets(in_col))


class WordNetGlossTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable):
    """
    PySpark Transformer that maps synsets to their respective definition (glosses)
    This transformer must be applied to a column consisting of Wordnet synset keys
    """
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        module = __import__("__main__")
        setattr(module, 'WordNetGlossTransformer', WordNetGlossTransformer)
        super(WordNetGlossTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def gloss(self, synset):
        return wn.synset(synset).definition()

    def _transform(self, dataset):
         # User defined function to map every word to a list of lemmas
        syn2gloss = udf(lambda syn: self.gloss(syn), ArrayType(StringType()))

        # Select the input column
        in_col = dataset[self.getInputCol()]

        # Get the name of the output column
        out_col = self.getOutputCol()

        #return dataset.withColumn(out_col, F.explode(text2lemmas(in_col)))
        return dataset.withColumn(out_col, syn2gloss(in_col))


"""
["I'll be going to the bank tomorrow", "Yesterday the bank was closed", "you should be able to go to the bank and solve the issue"]

["What about a walk along the river bank?", "today the banks look pretty dirty, I wonder what happened", "The forest extends across the banks of the old river]


"""