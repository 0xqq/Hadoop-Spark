from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

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
    .withColumn("BMI", data["Body mass index"] - 0)\
    .withColumn("Education", data["Education"] - 0)\
    .withColumn("hours", data["Absenteeism time in hours"] - 0)\
    .withColumn("child", data["Son"] - 0)\
    .withColumn("label", data["Seasons"] - 0)

data.show(5)

assem = VectorAssembler(inputCols=["label", "distance", "child"], outputCol='features')
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
evaluator = MulticlassClassificationEvaluator(labelCol="Education", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
print("Test set error = " + str(1 - accuracy))

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
