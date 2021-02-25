from numpy.lib.npyio import save
from src.dataset.dataset import DocumentCorpusDataset
from src.utils.utils import save_file, load_file

if __name__ == '__main__':

    dataset = DocumentCorpusDataset.from_tsv("../data/wiki-ja/ja.wikipedia_250k.txt")
    save_file(dataset, "../dataset/cached", "jp-wikipedia-dataset")
    sentences = list(dataset.sentences)
    
    print(sentences[:10])
