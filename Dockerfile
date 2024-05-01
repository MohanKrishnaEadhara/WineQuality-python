# Use an official Python runtime as the base image
FROM centos:7

RUN yum -y update && yum -y install python3 python3-devel python3-pip java-1.8.0-openjdk-devel wget

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas

# Set the working directory in the container
WORKDIR /app

# Install Java and Spark dependencies
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget && \
    wget https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz && \
    tar -xzf spark-3.2.0-bin-hadoop3.2.tgz && \
    mv spark-3.2.0-bin-hadoop3.2 /opt/spark && \
    rm spark-3.2.0-bin-hadoop3.2.tgz

# Set Spark environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AWS CLI
RUN pip install awscli

# Copy the PySpark application files to the working directory
COPY training.py .
COPY prediction.py .

# Set the entrypoint command to run your PySpark application
ENTRYPOINT ["spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:3.2.0", "training.py"]