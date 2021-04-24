from dataclasses import dataclass
import os
import random
from src.dataset.dataset import Dataset
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.tokenizers import JapaneseTokenizer
import json
from collections import Counter

tokenizer = JapaneseTokenizer()

@dataclass
class Document:
    title: str
    content: str
    label: int
    url: Optional[str]

    def __repr__(self) -> str:
        return f"title: {self.title}\ndocument: {self.content}"


class DocumentDataset(Dataset):
    def __init__(self, documents: List[Document], label_to_id: Dict[str, int]):
        super().__init__(documents)
        self.label_to_id = label_to_id

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i):
        return super().__getitem__(i)

    @classmethod
    def from_collection(cls, root_path, max_n_tokens=None):
        documents = []
        labels_to_id = {}
        for i, path in enumerate(os.listdir(root_path)):
            category = path
            print(f"Current category: {category} id: {i}")
            labels_to_id[category] = i
            collection_path = os.path.join(root_path, path)
            for doc in os.listdir(collection_path):
                doc_path = os.path.join(collection_path, doc)
                #print(f"Extracting documents from doc: {doc}")
                with open(doc_path, 'r', encoding='utf8') as f:
                    lines = f.readlines()
                    _, _, title = lines[0:3]
                    assert(title)
                    doc = '\n'.join(list(map(lambda line: line.strip(), lines[2:])))
                    example = Document(title, doc, i)
                    if max_n_tokens is not None:
                        splitted_documents = DocumentDataset.split_in_paragraphs(example, max_n_tokens=max_n_tokens)
                        documents.extend(splitted_documents)
                    else:
                        documents.append(
                            example
                        )
        random.shuffle(documents)
        return cls(documents, labels_to_id)

    @classmethod
    def from_json(cls, json_paths, max_n_tokens=None):
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        documents = []
        labels = set()
        labels_to_id = {}
        cat_id = 0
        urls = set()
        for json_path in json_paths:
            with open(json_path) as f:
                data = json.load(f)
                articles = data["articles"]
                for article in articles:
                    if hasattr(article, "url"):
                        url = article["url"]
                        if url in urls:
                            continue
                        urls.add(url)
                    else:
                        url = None
                    content = article["text"] 
                    if content is None or content == "":
                        continue
                    if isinstance(content, list):
                        content = content[0]
                    title = article["title"] 
                    if title is None:
                        title = ""
                    category = article["category"]
                    if category not in labels:
                        cat_id+=1
                        labels.add(category)
                        labels_to_id[cat_id-1] = category 
                    content = title + "\n" + content
                    document = Document(
                            url = url,
                            title=title,
                            content=content,
                            label=cat_id-1
                            )
                    if max_n_tokens is not None:
                            splitted_documents = DocumentDataset.split_in_paragraphs(document, max_n_tokens=max_n_tokens)
                            documents.extend(splitted_documents)
                    else:
                        documents.append(document)
        random.shuffle(documents)
        return cls(documents, labels_to_id)


    @staticmethod
    def split_in_paragraphs(document: Document, max_n_tokens=300) -> List[str]:
        splitted_documents = []
        tokens = tokenizer.tokenize(document.content)
        total_tokens = len(tokens)
        i = 0
        current_split_tokens = []
        for tok in tokens:
            i += 1
            if i >= max_n_tokens or i >= total_tokens:
                i = 0 
                paragraph = ' '.join(current_split_tokens)
                splitted_documents.append(
                    Document(
                        url = document.url,
                        title = document.title,
                        content = paragraph,
                        label = document.label
                    )
                )
                current_split_tokens = []
            else:
                current_split_tokens.append(tok)
        return splitted_documents

if __name__ == "__main__":

    categories = ["business", "culture", "economy", "politics", "society", "sports", "technology", "opinion", "local", "various"]
    paths = [f"../data/articles/nikkei/nikkei_{cat}.json" for cat in categories]
    documents = DocumentDataset.from_json(paths)
    print(f"Total number of documents: {len(documents)}")
    #print(f"Documents labels: {list(map(lambda doc: doc.label, documents))}")
    #print(f"Example documents: {documents[:3]}")
    label_to_cat = documents.label_to_id
    print(f"Mapping: {label_to_cat}")
    string_labels = list(map(lambda x: label_to_cat[x], documents.labels))
    print(f"Labels distribution: {Counter(string_labels)}")