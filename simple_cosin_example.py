from time import time

from pyspark import SparkContext
from spark_lp.text_rdd import TextRDD, TextsCorpus

if __name__ == "__main__":
    sc = SparkContext(appName="simpleCosinExample")
    main_path = '/home/alex/PycharmProjects/spark_lp/texts/'

    with open(f'{main_path}hamster0.txt') as f:
        hamster0 = f.read()
    with open(f'{main_path}hamster1.txt') as f:
        hamster1 = f.read()
    with open(f'{main_path}opera.txt') as f:
        opera0 = f.read()
    with open(f'{main_path}opera1.txt') as f:
        opera1 = f.read()

    corpus = TextsCorpus(sc, [hamster0, hamster1, opera0, opera1])
    print('Cosine similarity:')
    print('hamster0 vs hamster1: ', corpus.get_similarity(hamster0, hamster1))
    print('opera0 vs opera1: ', corpus.get_similarity(opera0, opera1))
    print('hamster0 vs opera0: ', corpus.get_similarity(hamster0, opera0))
