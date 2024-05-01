from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.hadoop:hadoop-aws:3.2.0 pyspark-shell'

# Clean the column names


def clean_column_name(col_name):
    return col_name.replace('"', '').strip()


def main():
    conf = SparkConf().setAppName("WineQualityTraining")
    spark = SparkSession.builder \
        .appName("WineQualityTraining") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .getOrCreate()

    try:
        # Load the training and validation datasets with semicolon delimiter
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

        # Prepare the feature columns
        featureColumns = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ]

        assembler = VectorAssembler(
            inputCols=featureColumns, outputCol="features")

        # String indexer for the label column
        indexer = StringIndexer(inputCol="quality", outputCol="label")

        # Logistic Regression model
        lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,
                                family="multinomial", labelCol="label", featuresCol="features")

        pipeline = Pipeline(stages=[assembler, indexer, lr])
        model = pipeline.fit(trainingData)

        # Make predictions on the validation dataset
        predictions = model.transform(validationData)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Accuracy =", accuracy)

        # Metrics
        print("Confusion matrix:")
        predictions.groupBy("label", "prediction").count().show()

        metrics = ["f1", "weightedPrecision", "weightedRecall"]
        for metric in metrics:
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                          metricName=metric)
            result = evaluator.evaluate(predictions)
            print(f"{metric} = {result}")

        # Save the model
        model.write().overwrite().save("s3a://wine-quality-dataset-bucket/WineQualityModel")

    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        # Clean up
        spark.stop()


if __name__ == "__main__":
    main()
