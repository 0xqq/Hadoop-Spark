from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os

from pyspark.mllib.evaluation import MulticlassMetrics

os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

# Load training data
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
data = spark.read.load("Absenteeism_at_work.csv", format="csv", header=True, delimiter=";")
data = data.withColumn("MOA", data["Month of absence"] - 0)\
    .withColumn("ROA", data["Reason for absence"] - 0)\
    .withColumn("distance", data["Distance from Residence to Work"] - 0)\
    .withColumn("BMI", data["Body mass index"] - 0)\
    .withColumn("Education", data["Education"] - 0)\
    .withColumn("hours", data["Absenteeism time in hours"] - 0)\
    .withColumn("child", data["Son"] - 0)\
    .withColumn("label", data["Seasons"] - 0)

data.show(5)

assem = VectorAssembler(inputCols=["label", "distance", "child"], outputCol='features')
data = assem.transform(data)

labelIndexer = StringIndexer(inputCol="Education", outputCol="indexedLabel").fit(data)

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test set accuracy = %g" % (accuracy))
print("Test set error = %g" % (1.0 - accuracy))

# Confusion Matrix, Precision and Recall
predictionAndLabels = predictions.select("prediction", "label").rdd
metrics = MulticlassMetrics(predictionAndLabels)

cm = metrics.confusionMatrix()
precision = metrics.precision()
recall = metrics.recall()
fmeasure = metrics.fMeasure()

print("\nConfusion Matrix\n" + str(cm))
print("\nPrecision\n" + str(precision))
print("\nRecall\n" + str(recall))
print("\nF-measure\n" + str(fmeasure))
