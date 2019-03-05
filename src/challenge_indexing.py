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
# For TF-IDF
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
# For combining counts across books
from pyspark.sql.functions import explode


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


# Indexing function which tokenizes terms and computes TF
def indexing_fun(**kwargs):
    # Load in each book and make each line a document
    book = sc.wholeTextFiles(file).map(lambda x: x[1])
    # Run the preprocess function on the book and zip
    book = book.mapPartitions(preproc).zipWithIndex()
    # Transform term tokens RDD to dataframe
    book = sqlContext.createDataFrame(book, ["list_of_words", "index"])
    # Calculate TF as raw count of terms
    cv = CountVectorizer(inputCol="list_of_words",
                         outputCol="raw_features")
    cvmodel = cv.fit(book)
    book = cvmodel.transform(book)
    # Combine book with corpus
    if 'corpus' in locals():
        book = book.union(corpus)
        book = book.select("raw_features", explode("list_of_words").alias("list_of_words")).groupBy("raw_features", "list_of_words")
        # Return book as corpus
        return book
    else:
        # Return book as corpus
        return book


# Calculate IDF and TF-IDF from full corpus
def tfidf_fun(corpus):
    # IDF
    idf = IDF(inputCol="raw_features",
              outputCol="features")
    idfModel = idf.fit(corpus)
    corpus = idfModel.transform(corpus)
    return corpus


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
    "setMaster": "local"
}


# Run pipeline functions
if __name__ == '__main__':
    [sc, sqlContext] = context(config)
    idx = 0
    for filename in os.listdir(os.getcwd()):
        file = mainpath + filename
        if idx == 0:
            corpus = indexing_fun(file=file, sc=sc, sqlContext=sqlContext)
        else:
            corpus = indexing_fun(file=file, sc=sc, sqlContext=sqlContext, corpus=corpus)
        idx = 1
    corpus = tfidf_fun(corpus)
