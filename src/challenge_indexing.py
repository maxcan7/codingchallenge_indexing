#! /usr/bin/env python


# Set up spark context and s3 bucket and folder config
def s3_to_pyspark(config):
    conf = SparkConf()
    conf.setMaster(config["publicDNS"])
    conf.setAppName("topicMakr")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    # Stopword list from nltk corpus
    StopWords = set(stopwords.words("english"))

    # Read text data from S3 Bucket
    tokens = []
    titles = []
    i = 0

    # Loop through all books in folder in bucket
    for file in filelist:
        # Load in all books and make each line a document
        books = sc.wholeTextFiles(file).map(lambda x: x[1])
        # Turn text to string for titles
        titles.append(books.collect()[0].encode('utf-8'))
        # Find 'Title:' in raw text
        start = titles[i].find('Title:')+7
        # Find the line break after the title
        end = titles[i][start:len(titles[i])].find('\r')
        end = start+end
        # Index title
        titles[i] = titles[i][start:end]
        # Run the preprocess function across books and zip
        tokens.append(books.mapPartitions(preproc).zipWithIndex())
        i += 1

    # Combine tokens
    tokens = sc.union(tokens)

    # Return variables for subsequent functions
    return sqlContext, tokens, titles


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

# Run pipeline functions
if __name__ == '__main__':
    