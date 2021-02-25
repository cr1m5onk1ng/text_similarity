from sentence_transformers import util

if __name__ == "__main__":

    nli_dataset_path = '../data/nli/all_nli.tsv.gz'
    sts_dataset_path = '../data/sts/stsbenchmark.tsv.gz'

    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)