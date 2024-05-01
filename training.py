from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.hadoop:hadoop-aws:3.2.0 pyspark-shell'


def clean_column_name(col_name):
    return col_name.replace('"', '').strip()


def main():
    conf = SparkConf().setAppName("WineQualityTraining")
    spark = SparkSession.builder \
        .appName("WineQualityTraining") \
        .config("spark.hadoop.version", "3.2.0") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    try:
        print("Loading datasets...")
        trainingData = spark.read.format("csv") \
            .option("header", "true") \
            .option("delimiter", ";") \
            .option("inferSchema", "true") \
            .load("s3a://wine-quality-dataset-bucket/TrainingDataset.csv")
        validationData = spark.read.format("csv") \
            .option("header", "true") \
            .option("delimiter", ";") \
            .option("inferSchema", "true") \
            .load("s3a://wine-quality-dataset-bucket/ValidationDataset.csv")

        trainingData = trainingData.toDF(
            *(clean_column_name(c) for c in trainingData.columns))
        validationData = validationData.toDF(
            *(clean_column_name(c) for c in validationData.columns))

        print("Assembling features...")
        featureColumns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                          "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                          "pH", "sulphates", "alcohol"]
        assembler = VectorAssembler(
            inputCols=featureColumns, outputCol="features")
        indexer = StringIndexer(inputCol="quality", outputCol="label")
        lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,
                                family="multinomial", labelCol="label", featuresCol="features")

        pipeline = Pipeline(stages=[assembler, indexer, lr])
        print("Training model...")
        model = pipeline.fit(trainingData)

        print("Evaluating model...")
        predictions = model.transform(validationData)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Accuracy =", accuracy)

        print("Saving model...")
        model.write().overwrite().save("s3a://wine-quality-dataset-bucket/WineQualityModel")
        print("Model saved successfully.")

    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
