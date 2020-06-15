from time import time

from pyspark import SparkContext, SQLContext
from pyspark.mllib.feature import HashingTF
from spark_lp.text_rdd import TextRDD, TextsCorpus
from spark_lp.text import Text
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType

if __name__ == "__main__":
    sc = SparkContext(appName="speedTest")
    with open('big_text.txt', 'r') as f:
        raw_text = f.read()

    init_start = time()
    text = TextRDD(sc, raw_text)
    split_start = time()
    text.split_to_sentences()
    sentences = text.sentences.collect()
    split_end = time()
    token_start = time()
    text.tokenize()
    all_words = text.words.collect()
    token_end = time()
    filter_start = time()
    text.filter_stop_words()
    text.words.collect()
    end = time()



    print("ВХІДНІ ДАНІ:")
    print("Кількість знаків: ", len(raw_text))

    print("ВИХІДНІ ДАНІ:")
    print("Загальний час обробки: ", end - init_start)
    print("Речень: ", len(sentences))
    print("Слів: ", sum([len(sent) for sent in sentences]))
    print("Загальний час на структуризацію: ", split_end - split_start)
    print("Загальний час на токенізацію: ", token_end - token_start)
    print("В середньому на слово: ", (token_end - token_start) / len(all_words))
    final_words = text.words.collect()
    filtered_words = len(all_words) - len(final_words)
    print("Відфільтровано слів: ", filtered_words)
    print("Час на фільтрацію: ", end - filter_start)
    print("В середньму на слово:", (end - filter_start) / filtered_words)
    print("Середній час повної обробки на слово: ",
          (end - init_start) / len(all_words))
