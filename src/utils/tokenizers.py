import MeCab
from typing import Any, Union
import nltk


class JapaneseTokenizer():
    def __init__(self):
        self.tokenizer = MeCab.Tagger('-Owakati')
        self.sentence_detector = nltk.RegexpTokenizer(u'[^　！？。]*[！？。.\n]')

    def tokenize(self, text: str):
        return self.tokenizer.parse(text).split()

    def split_text(self, text):
        return self.sentence_detector.tokenize(text)
