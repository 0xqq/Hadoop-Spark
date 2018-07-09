import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()

    input = sc.parallelize([19, 45, 31, 6, 12, 1, 14, 0])

    # Map Reduce and Sort
    # Map and Reduce functions are like word count program
    # Sort by Key is used to sort the resulting array

    output = input.map(lambda n: (n, 1))\
        .reduceByKey(lambda a, b: a + b)\
        .sortByKey()\
        .collect()

    # Collect in a python list
    sortedList = []
    for x in range(len(output)):
        sortedList.append(output[x][0])

    # Print the list
    print(sortedList)