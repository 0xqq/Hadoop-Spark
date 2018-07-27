from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

# Load training data
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
data = spark.read.load("Absenteeism_at_work.csv", format="csv", header=True, delimiter=";")
data = data.withColumn("MOA", data["Month of absence"] - 0)\
    .withColumn("ROA", data["Reason for absence"] - 0)\
    .withColumn("distance", data["Distance from Residence to Work"] - 0)\
    .withColumn("bmi", data["Body mass index"] - 0)\
    .withColumn("age", data["Age"] - 0)\
    .withColumn("wt", data["Weight"] - 0)\
    .withColumn("ht", data["Height"] - 0)\
    .withColumn("label", data["Seasons"] - 0)

data.show(5)

assem = VectorAssembler(inputCols=["age", "wt", "ht"], outputCol='features')
data = assem.transform(data)


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test set accuracy = " + str(accuracy))
print("Test set error = " + str(1.0 - accuracy))

treeModel = model.stages[2]
# summary only
# print(treeModel)
