"""
This file contains a simple LDA topic modeling pipeline using PySpark 
Source: https://github.com/maobedkova/TopicModelling_PySpark_SparkNLP/blob/master/Topic_Modelling_with_PySpark_and_Spark_NLP.ipynb
"""

import sparknlp
from pyspark.sql import functions as F
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from sparknlp.annotator import Normalizer
from sparknlp.annotator import LemmatizerModel
from sparknlp.annotator import StopWordsCleaner
from sparknlp.annotator import PerceptronModel
from sparknlp.annotator import NGramGenerator
from sparknlp.base import Finisher
from pyspark.sql import types as T
from nltk.corpus import stopwords

from pyspark.ml import Pipeline

LANG = "english"


spark = sparknlp.start()

path = 'Some path'
data = spark.read.csv(path, header=True)
text_col = 'sentences'
text_data = data.select(text_col).filter(F.col(text_col).isNotNull())

document_assembler = DocumentAssembler() \
    .setInputCol(text_col) \
    .setOutputCol("document")

tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokens')

normalizer = Normalizer() \
     .setInputCols(['tokens']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

stopwords = stopwords.words(LANG)

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('cleaned_lemmatized') \
     .setStopWords(stopwords)

ngrammer = NGramGenerator() \
    .setInputCols(['lemmatized']) \
    .setOutputCol('ngrams') \
    .setN(3) \
    .setEnableCumulative(True) \
    .setDelimiter('_')

pos_tagger = PerceptronModel.pretrained('pos_anc') \
     .setInputCols(['document', 'lemmatized']) \
     .setOutputCol('pos')

finisher = Finisher() \
     .setInputCols(['unigrams', 'ngrams', 'pos'])



pipeline = Pipeline() \
     .setStages([document_assembler,                  
                 tokenizer,
                 normalizer,                  
                 lemmatizer,                  
                 stopwords_cleaner, 
                 pos_tagger,
                 ngrammer,  
                 finisher])

processed_text = pipeline.fit(text_data).transform(text_data)

udf_join_arr = F.udf(lambda x: ' '.join(x), T.StringType())
processed_text  = processed_text.withColumn('finished_pos', udf_join_arr(F.col('finished_pos')))

pos_documentAssembler = DocumentAssembler() \
     .setInputCol('finished_pos') \
     .setOutputCol('pos_document')

pos_tokenizer = Tokenizer() \
     .setInputCols(['pos_document']) \
     .setOutputCol('pos')

pos_ngrammer = NGramGenerator() \
    .setInputCols(['pos']) \
    .setOutputCol('pos_ngrams') \
    .setN(3) \
    .setEnableCumulative(True) \
    .setDelimiter('_')

pos_finisher = Finisher() \
     .setInputCols(['pos', 'pos_ngrams']) 

pos_pipeline = Pipeline() \
     .setStages([pos_documentAssembler,                  
                 pos_tokenizer,
                 pos_ngrammer,  
                 pos_finisher])


processed_text = pos_pipeline.fit(processed_text).transform(processed_text)