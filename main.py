from pyspark import SparkContext, SQLContext
from spark_lp.text_rdd import TextRDD, TextsCorpus
from spark_lp.text import Text

if __name__ == "__main__":
    sc = SparkContext(appName="MyApp")
    sqlContext = SQLContext(sc)

    # extract and data from csv and preprocess it
    df = sqlContext.read.load('data.csv',
                              format='com.databricks.spark.csv',
                              header='true',
                              inferSchema='true')
    df.show()
    texts_rdd = df.select('Body').rdd.map(lambda row: row.Body)

    raw_text1, raw_text2, raw_text3 = texts_rdd.zipWithIndex().filter(
        lambda row: row[1] in [1, 2, 3]).map(lambda row: row[0]).collect()

    # work with document corpus
    doc_corpus = TextsCorpus(sc,
                             texts_rdd)  # tokenize all docs and compute TF-IDF
    print("TF-IDF for text1:",
          doc_corpus.get_tfidf(raw_text1))  # get tf_idf for certain text

    # get cosine similarity between 2 docs
    print("Similarity between text1 and text2:",
          doc_corpus.get_similarity(raw_text1, raw_text2))

    print("Similarity between text2 and text3:",
          doc_corpus.get_similarity(raw_text2, raw_text3))

    # work with separate text
    text = Text(raw_text3)  # TextRDD has the same functionality
    # structurize raw text to sentences and words
    text.split_to_sentences()
    print("Text3 structure:", text.sentences)

    # tokenize words and save info abot each word
    text.tokenize()
    print("Words info:", text.words_info)

    # remove stop words
    text.filter_stop_words()
    print("Filtered stop words:", text.sentences)
