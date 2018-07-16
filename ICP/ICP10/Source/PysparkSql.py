import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Pyspark SQL ICP3") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# 1 - Import the dataset and create data frames directly on import
df = spark.read.load("C:\\Users\\amehta\\PycharmProjects\\Spark\\ConsumerComplaints.csv",
                     format="csv", sep=",", inferSchema="true", header="true")
print("Step 1 - Import data successful")

# 2 - Save data to file
df.write.save("consumer_issue_company", format="csv", header="true")
print("Step 2 - Save data to file successful")

# 3 - Count the number of repeated record in the dataset
total = df.count()
unique = df.dropDuplicates().count()   # countDistinct() in scala
print("Step 3 - Number of repeated records = " + str(total-unique))

# 4 - Union and Order by company name
unionDf = df.union(df)   # union with self
print("Step 4.1 - Union Dataframe")
unionDf.show()

orderAscDf = df.orderBy(df['Company'])
print("Step 4.2 - Order by company asc")
orderAscDf.show()

orderDescDf = df.orderBy(df['Company'].desc())
print("Step 4.3 - Order by company desc")
orderDescDf.show()

# 5 - Group by zip code
groupDf = df.groupBy("Zip Code").count()
print("Step 5 - Group by zip code")
groupDf.show()

# 6 - Basic Queries: Joins and Aggregation
df.createOrReplaceTempView("complaints")
sqlDf = spark.sql("SELECT * FROM complaints")
sqlDf.show()

print("Step 6.1 - Join")
joinDf = df.crossJoin(df)
joinDf.show()

print("Step 6.2 - Aggregate")
countDf = spark.sql("SELECT COUNT(*) cnt FROM complaints WHERE Company = 'Bank of America'")
countDf.show()

# 7 - Fetch 13th row
print("Step 7 - Fetch 13th row")
print(df.take(13)[-1])

rowDf = spark.sql("SELECT * FROM complaints c LIMIT 13")
rowDf.show()








