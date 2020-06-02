import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Set, Tuple
from pymorphy2 import MorphAnalyzer
from pymorphy2.analyzer import Parse
import numpy as np
from pyspark.mllib.linalg import SparseVector

from spark_lp.choices import Lang
from spark_lp.stop_words import stop_words_dict

text_separators = '\s*[.!?]+\s*'
sent_separators = '[,:;\ \-—"]+'


def clean_text(text: str) -> str:
    text = re.sub('\n+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text


def split_to_sentences(text: str, is_cleaned=False) -> List[str]:
    if not is_cleaned:
        text = clean_text(text)
    return [sent for sent in re.split(text_separators, text) if sent]


def split_to_words(sentence: str) -> List[str]:
    return [word for word in re.split(sent_separators, sentence) if word]


def normalize_word(word: str, lang: Lang = Lang.UK, deep=False) -> str:
    morph = MorphAnalyzer(lang=lang.value)

    norm = morph.parse(word.lower())[0].normal_form
    if deep:
        while norm != word:
            word = norm
            norm = morph.parse(word.lower())[0].normal_form

    return norm


def normalize_sent(
        sentence: List[str], lang: Lang = Lang.UK, deep=False
) -> List[str]:
    return [normalize_word(word, lang, deep) for word in sentence]


def parse_sent(sentence: List[str], lang: Lang = Lang.UK) -> List[Parse]:
    morph = MorphAnalyzer(lang=lang.value)
    return [morph.parse(word)[0] for word in sentence]


def parse_obj_to_dict(parse_obj: Parse) -> dict:
    tag = parse_obj.tag
    return {
        'normal_form': parse_obj.normal_form,
        'pos': str(tag.POS),
        'case': str(tag.case),
        'number': str(tag.number),
        'gender': str(tag.gender)
    }


def tokenize_sent(sentence: List[str], lang: Lang = Lang.UK) -> List[dict]:
    morph = MorphAnalyzer(lang=lang.value)
    words = [morph.parse(word)[0] for word in sentence]
    words_info = []
    for word in words:
        word_info = parse_obj_to_dict(word)
        word_info['word'] = word.normal_form
        words_info.append(word_info)
    return words_info


def normalize_text(
        text: List[List[str]], lang: Lang = Lang.UK, deep=False
) -> List[List[str]]:
    return [normalize_sent(sentence, lang, deep) for sentence in text]


def filter_stop_words(
        sentence: List[str],
        stop_words: Union[None, Set[str]] = None,
        lang: Lang = None
) -> List[str]:
    if not stop_words:
        if not lang:
            raise ValueError('One of "stop_words" or "lang" is required')
        stop_words = get_stop_words(lang)
    return [word for word in sentence if word not in stop_words]


@lru_cache(maxsize=None)
def get_stop_words(language: Lang) -> Set[str]:
    return stop_words_dict[language]


def update_vectors(v1: SparseVector, v2: SparseVector) -> Tuple[list, list]:
    indices = list(set(v1.indices).union(set(v2.indices)))
    v11 = {index: 0 for index in indices}
    v22 = {index: 0 for index in indices}
    v11.update(dict(zip(v1.indices, v1.values)))
    v22.update(dict(zip(v2.indices, v2.values)))
    return list(v11.values()), list(v22.values())


def cos_sim(a: SparseVector, b: SparseVector):
    v1, v2 = update_vectors(a, b)
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
