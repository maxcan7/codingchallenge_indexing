#! /usr/bin/env python


# For reading data directory
import os
import sys
# For setting up pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
# For processing dataframes
import re as re
# For importing stopwords
from nltk.corpus import stopwords

# Set main path
mainpath = sys.argv[1]
os.chdir(mainpath)

# Stopword list from nltk corpus
StopWords = set(stopwords.words("english"))

# Set up spark context
def context(config):
    conf = SparkConf()
    conf.setMaster(config["setMaster"])
    conf.setAppName("indexing")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    return sc, sqlContext


# Indexing function which tokenizes and computes tf-idf
def indexing_fun(file, sc, sqlContext):
    # Load in all books and make each line a document
    book = sc.wholeTextFiles(file).map(lambda x: x[1])
    # Run the preprocess function across books and zip
    tokens = book.mapPartitions(preproc).zipWithIndex()
    # Return variables for subsequent functions
    return sqlContext, tokens


# Convert tokens to TF-IDF and run LDA model
def books_to_lda(ldaparam, sqlContext, tokens, titles):
    # Transform term tokens RDD to dataframe
    df_txts = sqlContext.createDataFrame(tokens, ["list_of_words", "index"])
    # Replace index to monotonically increasing set of values
    # (given zipWithIndex on tokens occurs over loop
    # and therefore is all zeros)
    df_txts = df_txts.withColumn('index', monotonically_increasing_id())
    # TF
    cv = CountVectorizer(inputCol="list_of_words",
                         outputCol="raw_features")
    cvmodel = cv.fit(df_txts)
    result_cv = cvmodel.transform(df_txts)

    # Create vocab list
    vocab = cvmodel.vocabulary

    # IDF
    idf = IDF(inputCol="raw_features",
              outputCol="features")
    idfModel = idf.fit(result_cv)
    result_tfidf = idfModel.transform(result_cv)


# Iterator function for processing partitioned data
def preproc(iterator):
    strip = lambda document: document.strip()
    lower = lambda document: document.lower()
    split = lambda document: re.split(" ", document)
    alpha = lambda word: [x for x in word if x.isalpha()]
    minlen = lambda word: [x for x in word if len(x) > 3]
    nostops = lambda word: [x for x in word if x not in StopWords]
    for y in iterator:
        # Remove leading and trailing characters
        y = strip(y)
        # Make lowercase
        y = lower(y)
        # Tokenize words (split) by space
        y = split(y)
        # Remove words that have non-English alphabet characters
        y = alpha(y)
        # Remove words of size less than 3
        y = minlen(y)
        # Remove words from the nltk corpus stop words list
        y = nostops(y)
        yield y


# Set configurations
config = {
    # Your Public DNS or local
    "setMaster": "local",
}


# Run pipeline functions
if __name__ == '__main__':
    [sc, sqlContext] = context(config)
    for filename in os.listdir(os.getcwd()):
        file = os.getcwd() + '\\' + filename
        indexing_fun(file, sc, sqlContext)

