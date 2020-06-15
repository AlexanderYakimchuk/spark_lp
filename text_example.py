from pyspark import SparkContext

from spark_lp import TextRDD

if __name__ == "__main__":
    sc = SparkContext(appName='Test')
    with open('hamster1.txt', 'r') as f:
        text_raw = f.read()

    text = TextRDD(sc, text_raw)
    text.process()
    print(text.words_info.collect())