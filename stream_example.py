from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from spark_lp.text_rdd import TextRDD, TextsCorpus, TextsStream


def print_text(text):
    m = text.collect()
    print('HERE')
    print(len(m))
    print(m)
    for t in m:
        print(t.tfidf)
        print(type(t.tfidf.indices))
        print(str(t.tfidf.indices))


if __name__ == '__main__':
    sc = SparkContext(appName="PythonStreamingWordCount")
    ssc = StreamingContext(sc, 10)
    lines = ssc.socketTextStream('0.0.0.0', 8080)
    corpus = TextsStream(sc, lines)
    corpus.texts.pprint()
    # corpus.tfidfs.pprint()
    # corpus.idf.pprint()
    corpus.windowed_texts.foreachRDD(print_text)
    ssc.start()
    ssc.awaitTermination()
