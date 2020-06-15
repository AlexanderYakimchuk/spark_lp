from pyspark import SparkContext, SQLContext
from pyspark.mllib.feature import HashingTF
from spark_lp.text_rdd import TextRDD, TextsCorpus
from spark_lp.text import Text
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType

sc = SparkContext(appName="MyApp")


def get_list_from_row(row):
    return [row.Id, row.Title, row.Body, row.Summary20, row.Cosine20,
            row.Summary40, row.Cosine40]


if __name__ == "__main__":
    sqlContext = SQLContext(sc)

    # extract and data from csv and preprocess it
    df = sqlContext.read.load('UA_dataset_summaries.csv',
                              format='com.databricks.spark.csv',
                              header='true',
                              inferSchema='true')
    valid = df.where(
        df["Cosine20"].cast("float").isNotNull() & df["Cosine40"].cast(
            "float").isNotNull())
    valid.show()
    df = valid
    print(df.collect())
    # df = valid
    texts = df.select('Body').rdd.map(lambda row: row.Body)
    texts = texts.union(
        df.select('Summary20').rdd.map(lambda row: row.Summary20))
    texts = texts.union(
        df.select('Summary40').rdd.map(lambda row: row.Summary40))
    corpus = TextsCorpus(sc, texts)
    # texts.show()
    sum_delta_20 = 0
    max_delta_20 = 0
    sum_rel_delta_20 = 0
    my_cosines_20 = []
    sum_delta_40 = 0
    max_delta_40 = 0
    sum_rel_delta_40 = 0
    my_cosines_40 = []
    for row in df.collect():
        # print(row.Title)
        my_cosine_20 = corpus.get_similarity(row.Body, row.Summary20)
        my_cosines_20.append(my_cosine_20)
        delta_20 = abs(my_cosine_20 - float(row.Cosine20))
        if delta_20 > max_delta_20:
            max_delta_20 = delta_20
        sum_delta_20 += delta_20
        sum_rel_delta_20 += delta_20 / float(row.Cosine20)
        my_cosine_40 = corpus.get_similarity(row.Body, row.Summary40)
        my_cosines_40.append(my_cosine_40)
        delta_40 = abs(my_cosine_40 - float(row.Cosine40))
        if delta_40 > max_delta_40:
            max_delta_40 = delta_40
        sum_delta_40 += delta_40
        sum_rel_delta_40 += delta_40 / float(row.Cosine40)
        # print(corpus.get_similarity(row.Body, row.Summary20), row.Cosine20)

    len_data = len(my_cosines_20)
    print('Cosine 20')
    print("Середня абсолютна похибка:", sum_delta_20 / len_data)
    print("Максимальна абсолютна похибка:", max_delta_20)
    print("Середня відносна похибка", sum_rel_delta_20 / len_data)
    print('Cosine 40')
    print("Середня абсолютна похибка:", sum_delta_40 / len_data)
    print("Максимальна абсолютна похибка:", max_delta_40)
    print("Середня відносна похибка", sum_rel_delta_40 / len_data)
    result_data = []
    rows = df.collect()
    for i in range(len(my_cosines_20)):
        row = rows[i]
        l = get_list_from_row(row)
        l = l[:5] + [my_cosines_20[i]] + l[5:] + [my_cosines_40[i]]
        result_data.append(l)
    # print(df.collect())
    heads = ['Id', 'Title', 'Body', 'Summary20', 'Cosine20', 'RecalcCosine20', 'Summary40',
             'Cosine40', 'RecalcCosine40']
    df = sqlContext.createDataFrame([l for l in result_data], heads)
    df.show()
    df.toPandas().to_csv('UA_dataset_recalc.csv')

    # def similarity(text1, text2):
    #     tf1 = HashingTF().transform(Text(text1).process().words)
    #     tf2 = HashingTF().transform(Text(text2).process().words)
    #     corpus = TextsCorpus(sc, [text1, text2])
    #     return corpus.get_similarity(text1, text2)
    #
    # udf_similarity = F.udf(similarity, FloatType())
    # data = valid.withColumn("MyCosine20",
    #                         udf_similarity("Body", "Summary20"))
    # data.show()

    # texts_rdd = df.select('Body').rdd.map(lambda row: row.Body)
    #
    # raw_text1, raw_text2, raw_text3 = texts_rdd.zipWithIndex().filter(
    #     lambda row: row[1] in [1, 2, 3]).map(lambda row: row[0]).collect()
    #
    # # work with document corpus
    # doc_corpus = TextsCorpus(sc,
    #                          texts_rdd)  # tokenize all docs and compute TF-IDF
    # print("TF-IDF for text1:",
    #       doc_corpus.get_tfidf(raw_text1))  # get tf_idf for certain text
    #
    # # get cosine similarity between 2 docs
    # print("Similarity between text1 and text2:",
    #       doc_corpus.get_similarity(raw_text1, raw_text2))
    #
    # print("Similarity between text2 and text3:",
    #       doc_corpus.get_similarity(raw_text2, raw_text3))
    #
    # # work with separate text
    # text = Text(raw_text3)  # TextRDD has the same functionality
    # # structurize raw text to sentences and words
    # text.split_to_sentences()
    # print("Text3 structure:", text.sentences)
    #
    # # tokenize words and save info abot each word
    # text.tokenize()
    # print("Words info:", text.words_info)
    #
    # # remove stop words
    # text.filter_stop_words()
    # print("Filtered stop words:", text.sentences)
