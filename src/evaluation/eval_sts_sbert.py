from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import gzip
import csv
from sentence_transformers.readers import InputExample
import logging

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open("../data/sts/stsbenchmark.tsv.gz", 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    evaluator(model=model, output_path="./results")