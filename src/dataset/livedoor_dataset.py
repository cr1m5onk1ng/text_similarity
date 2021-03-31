import os
from typing import List
import MeCab
import nltk

t = MeCab.Tagger('-Owakati')
sent_detector = nltk.RegexpTokenizer(u'[^　！？。]*[！？。.\n]')


def tokenize(text):
    return t.parse(text).split()

def split_text(text):
    return sent_detector.tokenize(text)

def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


class DocumentExample:
    def __init__(self, title, document):
        self.title = title
        self.document = document

    def __repr__(self) -> str:
        return f"title: {self.title}\ndocument: {self.document}"

    def split_in_paragraphs(self, max_tokens=300) -> List[str]:
        paragraphs = []
        tokens = tokenize(self.document)
        total_tokens = len(tokens)
        i = 0
        current_split_tokens = []
        for tok in tokens:
            i += 1
            if i >= max_tokens or i >= total_tokens:
                i = 0 
                paragraph = ' '.join(current_split_tokens)
                paragraphs.append(paragraph)
                current_split_tokens = []
            else:
                current_split_tokens.append(tok)
        return paragraphs

class LivedoorDataset:
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, i):
        return self.documents[i]

    @property
    def paragraphs(self):
        return flat_map(lambda doc: doc.split_in_paragraphs(), self.documents)

    @classmethod
    def from_collection(cls, root_path):
        documents = []
        for path in os.listdir(root_path):
            collection_path = os.path.join(root_path, path)
            for doc in os.listdir(collection_path):
                doc_path = os.path.join(collection_path, doc)
                print(f"Extracting documents from doc: {doc}")
                with open(doc_path, 'r', encoding='utf8') as f:
                    lines = f.readlines()
                    _, _, title = lines[0:3]
                    assert(title)
                    doc = '\n'.join(list(map(lambda line: line.strip(), lines[2:])))
                    example = DocumentExample(title, doc)
                    documents.append(example)
        return cls(documents)



if __name__ == "__main__":

    documents = LivedoorDataset.from_collection('../data/livedoor/text/')
    print(f"Total number of documents: {len(documents)}")
    paragraphs = documents.paragraphs
    print(f"Total number of paragraphs: {len(paragraphs)}")
    print(f"Example parapgraphs: {paragraphs[0:3]}")