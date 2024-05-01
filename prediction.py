from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.hadoop:hadoop-aws:3.2.0 pyspark-shell'


def main():
    conf = SparkConf().setAppName("WineQualityPrediction")
    spark = SparkSession.builder \
        .appName("WineQualityTraining") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    try:
        # Load the saved model
        model = PipelineModel.load(
            "s3a://wine-quality-dataset-bucket/WineQualityModel")

        # Load the test dataset
        testData = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("s3a://wine-quality-dataset-bucket/TestDataset.csv")

        if testData.rdd.isEmpty():
            raise ValueError(
                "No data loaded. Check if the file exists and is not empty.")

        print("Test data schema:")
        testData.printSchema()
        testData.show(5)

        # Make predictions on the test dataset
        predictions = model.transform(testData)

        print("Predictions schema:")
        predictions.printSchema()

        # Evaluate the model performance on the test dataset
        evaluator = MulticlassClassificationEvaluator() \
            .setLabelCol("quality") \
            .setPredictionCol("prediction") \
            .setMetricName("f1")

        # Print the F1 score on the test dataset
        f1Score = evaluator.evaluate(predictions)
        print("F1 score on test data:", f1Score)

    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
