from typing import Union, Set, List

from pyspark.mllib.feature import HashingTF, IDF, IDFModel
from pyspark.ml.feature import CountVectorizer
from pyspark import RDD, SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.sql import SQLContext, SparkSession

from spark_lp.choices import Lang
from spark_lp.itext import IText
from spark_lp.text import Text
from spark_lp.utils import split_to_sentences, split_to_words, normalize_sent, \
    filter_stop_words, parse_sent, parse_obj_to_dict, tokenize_sent, \
    get_stop_words, cos_sim
from langdetect import detect
from pyspark.sql.types import *
import networkx as nx


class TextRDD(IText):
    def __init__(self, sc: SparkContext, text: str):
        self.text: str = text
        self._origin_sents: Union[RDD, None] = None
        self._sents: Union[RDD, None] = None
        self._words_info: Union[RDD, None] = None
        self._words: Union[RDD, None] = None
        self._tf = None
        self.idf: Union[IDFModel, None] = None
        self._tfidf = None
        self.sc: SparkContext = sc
        self.spark = SparkSession(self.sc)
        lang = detect(text)
        if lang not in ['uk', 'ru']:
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
        sents = self.sc.parallelize(
            split_to_sentences(self.text, is_cleaned=False))
        self._origin_sents = sents
        self._sents = sents.map(split_to_words)

    def tokenize(self):
        lang = self.lang
        self._sents = self._sents.map(
            lambda sent: normalize_sent(sent, lang))
        self._words_info = self._sents.flatMap(
            lambda sent: tokenize_sent(sent, lang))
        self._words = self._words_info.map(
            lambda word_info: word_info['normal_form'])

    def filter_stop_words(self, stop_words=None):
        lang = self.lang
        stop_words = stop_words or get_stop_words(self.lang)
        self._sents = self._sents.map(
            lambda sent: filter_stop_words(sent, stop_words, lang))
        self._words = self._words.filter(
            lambda word: word not in stop_words
        )

    def process(self):
        self.split_to_sentences()
        self.tokenize()
        self.filter_stop_words()
        return self

    def sumarize(self):
        vertices = self._sents.zipWithIndex()
        vertices_df = vertices.toDF(['words', 'id'])
        pairs = vertices.cartesian(vertices).filter(
            lambda pair: pair[0][1] < pair[1][1])
        self.tfidf()
        tfidfs = self._tfidf
        edges = pairs.map(lambda pair: (
            pair[0][1], pair[1][1], cos_sim(tfidfs[pair[0][1]],
                                            tfidfs[pair[1][1]])
        ))
        g = nx.Graph()
        g.add_weighted_edges_from(edges.collect())
        pr = nx.pagerank(g)
        res = sorted(((i, pr[i], s) for i, s in enumerate(self._origin_sents.collect()) if i in pr),
               key=lambda x: pr[x[0]], reverse=True)
        print('\n'.join([str(r) for r in res]))
        # edges_df = edges.toDF(['src', 'dst', 'weight'])
        #
        # graph = GraphFrame(vertices_df, edges_df)
        # ranked_sents = graph.pageRank(resetProbability=0.15, tol=0.01)
        # print(vertices.collect())
        # ranked_sents.vertices.show(truncate=False)
        # print(ranked_sents.vertices.select(['id', 'pagerank']).rdd.sortBy(lambda row: row.pagerank).collect())

    def tfidf(self):
        tf = HashingTF().transform(self._sents)
        self._tf = tf
        tf.cache()
        idf = IDF().fit(tf)
        self.idf = idf
        tfidf = idf.transform(tf)
        self._tfidf = dict(enumerate(tfidf.collect()))

    @staticmethod
    def get_tfidf(idf, sentence: List[str]) -> SparseVector:
        tf = HashingTF().transform(sentence)
        return idf.transform(tf)


class TextsCorpus:
    def __init__(self, sc: SparkContext, texts: Union[List, RDD]):
        self.sc: SparkContext = sc
        self.texts: RDD = self._tokenize_texts(sc, texts)
        self.idf = self._compute_idf(self.texts)

    @staticmethod
    def _tokenize_texts(sc: SparkContext, texts: Union[List[str], RDD]):
        if isinstance(texts, list):
            return sc.parallelize([Text(text).process().words for text in texts])
        else:
            return texts.map(lambda text: Text(text).process().words)

    @staticmethod
    def _compute_idf(texts: RDD) -> IDFModel:
        tf = HashingTF().transform(texts)
        tf.cache()
        idf = IDF().fit(tf)
        return idf

    def get_tfidf(self, text_str) -> SparseVector:
        tf = HashingTF().transform(Text(text_str).process().words)
        return self.idf.transform(tf)

    def extend(self, texts: List[str]):
        self.texts = self.texts.union(self._tokenize_texts(self.sc, texts))
        self.idf = self._compute_idf(self.texts)

    def get_similarity(self, text1: str, text2: str) -> float:
        v1 = self.get_tfidf(text1)
        v2 = self.get_tfidf(text2)
        return cos_sim(v1, v2)
