from typing import Union, List

from langdetect import detect

from spark_lp.choices import Lang
from spark_lp.itext import IText
from spark_lp.utils import split_to_sentences, split_to_words, normalize_sent, \
    filter_stop_words, parse_sent, parse_obj_to_dict


class Text(IText):
    def __init__(self, text: str):
        self.text: str = text
        self._sents: Union[List[str], None] = None
        self._words_info: Union[dict, None] = None
        self._words: Union[list, None] = None
        lang = detect(text)
        if lang not in ('uk', 'ru'):
            lang = 'uk'
        self.lang: Lang = Lang(lang)

    @property
    def sentences(self):
        return self._sents

    @property
    def words_info(self):
        return self._words_info

    @property
    def words(self):
        return self._words

    def split_to_sentences(self):
        sents = split_to_sentences(self.text, is_cleaned=False)
        self._sents = [split_to_words(sent) for sent in sents]

    def tokenize(self):
        normalized_sents = []
        words = []
        words_info = {}
        for sent in self._sents:
            tokens = parse_sent(sent, self.lang)
            normalized_words = []
            for token in tokens:
                normalized_words.append(token.normal_form)
                words_info[token.word] = parse_obj_to_dict(token)
            words.extend(normalized_words)
            # words.extend(normalized_words)
            normalized_sents.append(normalized_words)

        self._sents = normalized_sents
        self._words = words
        self._words_info = words_info

    def filter_stop_words(self, stop_words=None):
        self._sents = [filter_stop_words(sent, stop_words, self.lang) for sent
                       in self._sents]
        self._words = filter_stop_words(self._words, stop_words, self.lang)

    def process(self):
        self.split_to_sentences()
        self.tokenize()
        self.filter_stop_words()
        return self
