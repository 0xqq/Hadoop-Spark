import os
os.environ["SPARK_HOME"] = "C:\spark"
os.environ["HADOOP_HOME"] = "C:\winutils"

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)  # IndexRowMatrix

    # Method 1:
    m = sc.parallelize([[2, 2, 2], [3, 3, 3]]).zipWithIndex()
    n = sc.parallelize([[1, 1], [4, 4], [3, 3]]).zipWithIndex()

    # Create an Index Row Matrix
    # Convert to Block Matrix
    mat1 = IndexedRowMatrix(m.map(lambda row: IndexedRow(row[1], row[0]))).toBlockMatrix()
    mat2 = IndexedRowMatrix(n.map(lambda row2: IndexedRow(row2[1], row2[0]))).toBlockMatrix()

    # Method 2:
    #mat1 = BlockMatrix(m, 2, 3)
    #mat2 = BlockMatrix(n, 3, 2)

    # Use of multiply function from pyspark.mllib.linalg
    mat_mul_output = mat1.multiply(mat2).toLocalMatrix()
    print(mat_mul_output)