from dataclasses import dataclass
from src.utils.wikipedia_extractor import WikipediaExtractor
from src.dataset.documents_dataset import Document, DocumentDataset
from src.dataset.dataset import CrossValidationDataset, Dataset
from typing import Dict, List, Set, Union
import json
from tqdm import tqdm
import random
from src.utils import utils

CATEGORIES = ["スポーツ", "旅行", "ワーク", "芸術", "マスメディア", "教育", "娯楽", "政府", "レジャー活動", "舞台芸術", "フィットネス", "政治", "ゲーミング", 
                    "宗教", "交通", "戦争", "科学", "哲学の文献", "社会", "ビジネス", "健康", "技術"]

CATEGORIES_EN = ["Sports", "Travel", "Work", "The arts", "Mass media", "Education", "Entertainment", "Government", "Leisure activities", "Performing arts", "Physical exercise", "Politics",
                    "Gaming", "Religion", "Transport", "War", "Science", "Philosophical literature", "Society", "Business", "Health", "Technology", ""]

@dataclass
class WikipediaDocument(Document):
    id: str


class WikipediaDataset(DocumentDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i):
        return super().__getitem__(i)

    @staticmethod
    def split_in_paragraphs(document: WikipediaDocument, max_n_tokens=300) -> List[str]:
        splitted_documents = []
        tokens = utils.tokenize(document.content)
        total_tokens = len(tokens)
        i = 0
        current_split_tokens = []
        for tok in tokens:
            i += 1
            if i >= max_n_tokens or i >= total_tokens:
                i = 0 
                paragraph = ' '.join(current_split_tokens)
                splitted_documents.append(
                    WikipediaDocument(
                        id = document.id,
                        title = document.title,
                        content = paragraph,
                        label = document.label
                    )
                )
                current_split_tokens = []
            else:
                current_split_tokens.append(tok)
        return splitted_documents

    @property
    def train_split(self):
        return Dataset(self.examples, self.labels)

    @property
    def test_split(self):
        return Dataset(self.test_portion, self.test_labels)

    @classmethod
    def from_collection(cls, files: List[str], page_ids: Union[Dict[str, str], None]=None, max_n_docs=None, max_n_tokens=None):
        cat_to_label = {}
        for i in range(len(CATEGORIES)):
            cat_to_label[CATEGORIES[i]] = i 
        documents = []
        n = 0
        print("Extracting wikipedia articles. This may take a while")
        iterator = tqdm(files, total=len(files))
        for file_name in iterator:
            with open(file_name, "r", encoding="utf8") as f:
                lines = f.readlines()
                for line in lines:
                    article = json.loads(line)
                    id = str(article["id"])
                    title = str(article["title"])
                    text = str(article["text"])
                    category = None
                    if page_ids is not None:
                        if id not in page_ids:
                            continue
                        else:
                            category = page_ids[id]
                    n += 1
                    if max_n_docs is not None and n >= max_n_docs:
                        break

                    document = WikipediaDocument(
                            id = id,
                            title = title,
                            content=text,
                            label=cat_to_label[category]
                        )
                    
                    if max_n_tokens is not None:
                        splitted_documents = WikipediaDataset.split_in_paragraphs(document, max_n_tokens=max_n_tokens)
                        documents.extend(splitted_documents)
                    else:
                        documents.append(
                            document
                        )
        random.shuffle(documents)
        return cls(documents, cat_to_label)


if __name__ == "__main__":
    url = "https://ja.wikipedia.org/w/api.php?"
    lang = "ja"
    extractor = WikipediaExtractor(
        url = url,
        lang = lang
    )
    print("Searching page ids...")
    ids = extractor.extract_ids_from_categories(CATEGORIES, max_pages=500)
    print("Done.")
    print()
    print("Building dataset.")
    files = list(utils.search_files("../data/wikipedia-dump/japanese/extracted/"))
    dataset = WikipediaDataset.from_collection(files, page_ids=ids, max_n_docs=None, max_n_tokens=64)
    print(f"Dataset size: {len(dataset)}")

    cross_val = CrossValidationDataset.create_folds(dataset, 7)

    print(f"Train splits: {cross_val.train_splits}")
    print(f"Test splits: {cross_val.test_splits}")