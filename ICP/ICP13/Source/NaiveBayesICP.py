from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

# Load training data
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Load data
data = spark.read.load("Absenteeism_at_work.csv", format="csv", header=True, delimiter=";")

# Columns renamed and value converted from string to int
data = data.withColumn("MOA", data["Month of absence"] - 0)\
    .withColumn("ROA", data["Reason for absence"] - 0)\
    .withColumn("distance", data["Distance from Residence to Work"] - 0)\
    .withColumn("bmi", data["Body mass index"] - 0)\
    .withColumn("age", data["Age"] - 0)\
    .withColumn("wt", data["Weight"] - 0)\
    .withColumn("ht", data["Height"] - 0)\
    .withColumn("label", data["Seasons"] - 0)

data.show(5)

# List of features(input variables)

#assem = VectorAssembler(inputCols=["age", "wt", "ht"], outputCol='features')
assem = VectorAssembler(inputCols=["bmi"], outputCol='features')
data = assem.transform(data)

# Split the data into train and test
splits = data.randomSplit([0.7, 0.3], seed=42)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show(5)

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
print("Test set error = " + str(1 - accuracy))