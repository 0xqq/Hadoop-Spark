import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"
from pyspark import SparkContext


def mapper(x):
    x = x.split(" -> ")
    f1 = x[0]
    friends = x[1].split(" ")
    keys = []
    for f2 in friends:
        keys.append((''.join(sorted(f1 + f2)), friends))
    return keys


if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    # Load data
    lines = sc.textFile("C:\\Users\\amehta\\PycharmProjects\\Spark\\friendsInput", 1)

    # Logic
    line = lines.flatMap(mapper)\
                .reduceByKey(lambda x,y : list(set(x).intersection(y)))

    # Store output
    line.saveAsTextFile("mutualfriends")
    sc.stop()