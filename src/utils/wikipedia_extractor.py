from dataclasses import dataclass
from typing import Dict, List
from mediawiki import MediaWiki
import os
import os.path
import re
import bz2
import time


# Program version
__version__ = '3.0.5'

# ----------------------------------------------------------------------
# READER

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')
#tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>([^<]*)')
#                    1     2            3

def process_data(input_file, id, templates=False):
    """
    :param input_file: name of the wikipedia dump file.
    :param id: article id
    """

    if input_file.lower().endswith(".bz2"):
        input = bz2.open(input_file, mode='rt', encoding='utf-8')
    else:
        input = open(input_file)

    page = []
    for line in input:
        line = line
        if '<' not in line:         # faster than doing re.search()
            if page:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            page.append(line)
            inArticle = False
        elif tag == 'id':
            curid = m.group(3)
            if id == curid:
                page.append(line)
                inArticle = True
            elif not inArticle and not templates:
                page = []
        elif tag == 'title':
            if templates:
                if m.group(3).startswith('Template:'):
                    page.append(line)
                else:
                    page = []
            else:
                page.append(line)
        elif tag == '/page':
            if page:
                page.append(line)
                print(''.join(page))
                if not templates:
                    break
            page = []
        elif page:
            page.append(line)

    input.close()


@dataclass
class QueryParams:
    def __init__(self, category, cmlimit):
        self.category = category
        self.cmlimit = cmlimit

    @property
    def request(self):
        return {
            "action": "query",
            "list": "categorymembers", 
            "cmtitle": f"Category: {self.category}", "cmlimit": f"{self.cmlimit}"}


@dataclass
class WikipediaResponse:
    id: str
    title: str


class WikipediaExtractor:
    def __init__(self, lang: str, url: str):
        self.extractor = MediaWiki(lang=lang, url=url)

    def _map_document_to_response(self, json_response):
        return WikipediaResponse(
            id = str(json_response["pageid"]),
            title = str(json_response["title"])
        )

    def _map_documents_to_response(self, documents_list) -> List[WikipediaResponse]:
        documents = list(map(lambda doc: self._map_document_to_response(doc), documents_list))
        assert documents
        return documents

    def extract_page(self, page_id: str, dump_file_path: str):
        process_data(dump_file_path, page_id, True)

    def extract_ids_from_category(self, category, max_pages) -> List[str]:
        params = QueryParams(
            category = category,
            cmlimit = max_pages
        )
        json_response = self.extractor.wiki_request(params.request)
        documents_list = json_response["query"]["categorymembers"]
        docs =  self._map_documents_to_response(documents_list)
        ids = list(map(lambda doc: doc.id, docs))
        assert ids
        return ids

    def extract_ids_from_categories(self, categories: List[str], max_pages: int) -> Dict[str, str]:
        ids_to_cat = {}
        for cat in categories:
            ids = self.extract_ids_from_category(cat, max_pages=max_pages)
            print(f"Category {cat} number of pages: {len(ids)}")
            for id in ids:
                ids_to_cat[id] = cat
            time.sleep(1)
        return ids_to_cat
        
if __name__ == "__main__":

    url = "https://ja.wikipedia.org/w/api.php?"
    lang = "ja"
    extractor = WikipediaExtractor(
        url = url,
        lang = lang
    )

    extractor.extract_page("2444129", "jawiki-20210320-pages-articles-multistream.xml.bz2")

