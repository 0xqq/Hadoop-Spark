import os
os.environ["SPARK_HOME"] = "C:\\spark"
os.environ["HADOOP_HOME"] = "C:\\winutils"

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working thread and batch interval of 3 second
sc = SparkContext("local[2]", "TwitterWordCount")
ssc = StreamingContext(sc, 5)

from collections import namedtuple
fields = ("tag", "count" )
Tweet = namedtuple( 'Tweet', fields )

# Create a DStream that will connect to hostname:port, like localhost:9999
lines = ssc.socketTextStream("localhost", 5555)

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word length in each batch
pairs = words.map(lambda word: (word.lower(), 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)\
    .map( lambda rec: Tweet( rec[0], rec[1]))

# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.pprint()

ssc.start()  # Start the computation
ssc.awaitTermination()