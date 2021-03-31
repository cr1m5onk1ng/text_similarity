from dataclasses import dataclass
import os
from src.dataset.dataset import Dataset
from typing import Dict, List, Set, Union
import json
from tqdm import tqdm
from src.utils.utils import search_files
from src.utils.wikipedia_extractor import WikipediaExtractor
import random

CATEGORIES = ["スポーツ", "旅行", "ワーク", "芸術", "マスメディア", "教育", "娯楽", "政府", "レジャー活動", "舞台芸術", "フィットネス", "政治", "ゲーミング", 
                    "宗教", "交通", "戦争", "科学", "哲学の文献", "社会", "ビジネス", "健康", "貨幣", "技術"]

@dataclass
class WikipediaDocument:
    id: str
    title: str
    content: str
    category: str

class WikipediaDataset(Dataset):
    def __init__(self, test_split, test_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_split = test_split
        self.test_labels = test_labels

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, i):
        return self.documents[i]

    def get_by_category(self, category):
        return list(map(lambda doc: doc.category==category, self.documents))

    @property
    def train_split(self):
        return Dataset(self.examples, self.labels)

    @property
    def test_split(self):
        return Dataset(self.test_split, self.test_labels)

    @classmethod
    def from_collection(cls, files, page_ids: Union[Dict[str, str], None]=None, max_n_docs=None, test_perc=0.1):
        cat_to_label = {}
        for i in range(CATEGORIES):
            cat_to_label[CATEGORIES[i]] = i 
        documents = []
        labels = []
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
                    labels.append(cat_to_label[category])
                    n += 1
                    if n >= max_n_docs:
                        break
                    documents.append(
                        WikipediaDocument(
                            id = id,
                            title = title,
                            content=text,
                            category=category
                        )
                    )
        random.shuffle(documents)
        portion = int(len(documents) * test_perc)
        train_split = documents[:portion]
        test_split = documents[portion:]
        train_labels = labels[:portion]
        test_labels = labels[portion:]
        return cls(examples = train_split, test_split = test_split, labels = train_labels, test_labels = test_labels)


