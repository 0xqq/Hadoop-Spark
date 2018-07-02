import os

os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"]="C:\winutils"
from operator import add

from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    lines = sc.textFile("input.txt", 1)

    # Transformations - flatMap, map
    # Actions - reduceByKey, saveAsTextFile
    counts = lines.flatMap(lambda x: x.split(' ')) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(add)

    counts.saveAsTextFile("output")
    sc.stop()
