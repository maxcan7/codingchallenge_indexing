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
# For retrieving counts from count vectors
from pyspark.sql.functions import explode
from pyspark.sql.functions import when
from pyspark.sql.functions import lit

# Set main path
mainpath = sys.argv[1]
datapath = mainpath+'data/'
os.chdir(datapath)

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
    book = book.withColumn("index", when(book["index"] == 0, doc))
    docCounts = book.select("index", explode("list_of_words").alias("words")).groupBy("index", "words").count()
    # Write outputs
    book.select('list_of_words', 'index', 'raw_features').write.save('countvec_'+doc+'.parquet')
    docCounts.select('index', 'words', 'count').write.save('termdoccounts_'+doc+'.parquet')


# Calculate IDF and TF-IDF from full corpus
def tfidf_fun():
    # Load each file and do a running count of all terms for calculating TFIDF
    totCounts = []
    for filename in os.listdir(os.getcwd()):
        file = datapath + filename
        docCounts = spark.read.load(mainpath+'termdoccounts_'+filename+'.parquet')
        if not totCounts:
            # Add a column for words where each row is a unique word
            totCounts = docCounts.select('words')
            # Add a column for running count of number of documents each word appears in
            totCounts = totCounts.withColumn('docCount', lit(1))
        else:
            # Find words present in totCounts and current book and update count
            updatewords = totCounts.select('words').intersect(docCounts.select('words'))
            tmp = totCounts.withColumn('docCount', when(totCounts.words.alias('old') == updatewords.words.alias('new'), totCounts.docCount + 1).otherwise(totCounts.docCount))
            tmp2 = totCounts.select('words', (totCounts.words.alias('old') == updatewords.words.alias('new')).alias('update'))
            # Subtract from docCounts words already present in totCounts
            newwords = docCounts.select('words').subtract(totCounts.select('words')).withColumn('docCount', lit(1))
            # Add new words to totCounts
            totCounts = totCounts.unionAll(newwords)



    # IDF
    idf = IDF(inputCol="raw_features",
              outputCol="features")
    idfModel = idf.fit(book)
    tfidf_vecs = idfModel.transform(book)


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
    for filename in os.listdir(os.getcwd()):
        file = datapath + filename
        indexing_fun(file=file, sc=sc, sqlContext=sqlContext, doc=filename)
    tfidx_vecs = tfidf_fun()
